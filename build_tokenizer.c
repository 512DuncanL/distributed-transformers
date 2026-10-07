#include "tokenizer.h"

#define MAX_STORY_LEN 1024

/* ------------------------------------------------------------------ */
/*  init_tokenizer: set up the special tokens                         */
/* ------------------------------------------------------------------ */
void init_tokenizer(Tokenizer *t) {
    /* Order is significant: these IDs are relied on elsewhere (see TOK_* in
     * tokenizer.h; encode() emits TOK_BOS / TOK_EOS). Keep in sync. */
    const char *special_tokens[NUM_SPECIAL] = {
        "<EOS>",  /* TOK_EOS = 0 */
        "<PAD>",  /* TOK_PAD = 1 */
        "<BOS>",  /* TOK_BOS = 2 */
    };

    memset(t, 0, sizeof(Tokenizer));
    t->vocab_size = NUM_SPECIAL;

    for (int i = 0; i < NUM_SPECIAL; i++) {
        strcpy(t->tokens[i], special_tokens[i]);
    }
}

/* ------------------------------------------------------------------ */
/*  build_base_vocab: seed all 256 byte values as base tokens         */
/* ------------------------------------------------------------------ */
void build_base_vocab(Tokenizer *t) {
    /* Every byte gets a base token at id = NUM_SPECIAL + byte, so any input
     * is representable and there is no UNK fallback. The token string is the
     * raw byte; serialization escapes it as needed. Byte 0x00 is the one hole
     * a NUL-terminated C string can't hold (stored as ""), but text data never
     * contains it. */
    for (int b = 0; b < 256; b++) {
        t->tokens[NUM_SPECIAL + b][0] = (char)b;
        t->tokens[NUM_SPECIAL + b][1] = '\0';
    }
    t->vocab_size = BASE_VOCAB;

    printf("Base vocabulary: %d byte tokens\n", 256);
}

/* ------------------------------------------------------------------ */
/*  build_tokenizer: BPE training loop                                */
/* ------------------------------------------------------------------ */
void build_tokenizer(Tokenizer *t, char **rows, int row_count) {
    /* Allocate pair counting array on the heap (64MB) */
    int (*pair_counts)[MAX_VOCAB] = calloc(MAX_VOCAB, MAX_VOCAB * sizeof(int));
    if (!pair_counts) {
        fprintf(stderr, "Failed to allocate pair_counts (%dMB)\n",
                (int)(MAX_VOCAB * MAX_VOCAB * sizeof(int) / (1024 * 1024)));
        exit(1);
    }

    /* Track which rows have been touched this iteration (avoids full memset) */
    int *dirty_rows = calloc(MAX_VOCAB, sizeof(int));
    int *row_marked = calloc(MAX_VOCAB, sizeof(int));
    int dirty_count = 0;
    if (!dirty_rows || !row_marked) {
        fprintf(stderr, "Failed to allocate dirty tracking arrays\n");
        exit(1);
    }

    /* Allocate per-row token ID sequences with lengths */
    int **seqs = malloc(row_count * sizeof(int *));
    int  *seqlens = malloc(row_count * sizeof(int));
    if (!seqs || !seqlens) {
        fprintf(stderr, "Failed to allocate seqs\n");
        exit(1);
    }

    /* Allocate a working sequence buffer for pair replacement */
    int *seq = malloc(MAX_STORY_LEN * sizeof(int));
    if (!seq) {
        fprintf(stderr, "Failed to allocate sequence buffer\n");
        exit(1);
    }
    (void)seq;  /* in-place replacement, buffer kept for potential use */

    /* Convert all stories to token ID sequences */
    fprintf(stderr, "Converting %d stories to sequences...\n", row_count);
    fflush(stderr);
    for (int r = 0; r < row_count; r++) {
        int seq_len = 0;
        const char *s = rows[r];
        int prev_id = -1;

        while (*s) {
            if (s[0] == '<' && s[1] == 'E' && s[2] == 'N' && s[3] == 'D') {
                s += 4;
                prev_id = -1;
                continue;
            }
            int cur_id = NUM_SPECIAL + (unsigned char)*s;
            if (prev_id >= 0) {
                /* No special pair filtering here — we count in the loop below */
            }
            seq[seq_len++] = cur_id;
            prev_id = cur_id;
            s++;
        }

        seqs[r] = malloc(seq_len * sizeof(int));
        if (seqs[r]) memcpy(seqs[r], seq, seq_len * sizeof(int));
        seqlens[r] = seq_len;
    }

    int merges_done = 0;

    int original_vocab_size = t->vocab_size;
    for (int merge_step = 0; merge_step < MAX_VOCAB - original_vocab_size; merge_step++) {
        if (merge_step == 0) {
            fprintf(stderr, "Starting BPE loop...\n");
            fflush(stderr);
        }
        /* Step 1: recount adjacent pairs from the current sequences and track
         * the most frequent pair inline. A pair's count only grows during the
         * pass, so the running max settles on the true winner by the end — no
         * separate O(vocab^2) scan needed.
         * Zero only the rows touched last iteration (dirty_rows), so clearing
         * costs O(touched_rows * vocab) instead of memset-ing the full 64MB. */
        for (int i = 0; i < dirty_count; i++) {
            int a = dirty_rows[i];
            memset(pair_counts[a], 0, MAX_VOCAB * sizeof(int));
            row_marked[a] = 0;
        }
        dirty_count = 0;

        int best_a = 0, best_b = 0, best_count = 0;
        for (int r = 0; r < row_count; r++) {
            if (!seqs[r]) continue;
            int len = seqlens[r];
            for (int i = 0; i < len - 1; i++) {
                int a = seqs[r][i];
                int b = seqs[r][i + 1];
                if (!row_marked[a]) {
                    /* First pair starting with `a` this iteration — track the
                     * row so Step 1 clears it next iteration. */
                    dirty_rows[dirty_count++] = a;
                    row_marked[a] = 1;
                }
                int count = ++pair_counts[a][b];
                if (count > best_count) {
                    best_count = count;
                    best_a = a;
                    best_b = b;
                }
            }
        }

        /* If no pair appears more than once, we're done */
        if (best_count < 1) {
            printf("No more pairs to merge at step %d\n", merge_step);
            break;
        }

        /* Step 3: create new token = concatenation of the merged pair.
         * Copy the sources to locals first — the destination is in the same
         * t->tokens array and snprintf forbids overlapping src/dst. snprintf
         * truncates safely at MAX_TOKEN_LEN; the slot was zero-filled by
         * init_tokenizer and is written exactly once, so the bytes after the
         * NUL stay zero, keeping the fixed-block binary records deterministic. */
        int new_id = t->vocab_size;
        char a[MAX_TOKEN_LEN], b[MAX_TOKEN_LEN];
        strcpy(a, t->tokens[best_a]);
        strcpy(b, t->tokens[best_b]);
        snprintf(t->tokens[new_id], MAX_TOKEN_LEN, "%s%s", a, b);

        /* Step 4: replace every (A, B) occurrence with the new token, in place.
         * pair_counts is fully recomputed at the top of the next iteration, so
         * there are no incremental count updates to keep in sync here. */
        for (int r = 0; r < row_count; r++) {
            if (!seqs[r]) continue;
            int len = seqlens[r];
            int write = 0;
            for (int read = 0; read < len; read++) {
                if (read < len - 1 && seqs[r][read] == best_a && seqs[r][read + 1] == best_b) {
                    seqs[r][write++] = new_id;
                    read++;  /* skip B */
                } else {
                    seqs[r][write++] = seqs[r][read];
                }
            }
            seqlens[r] = write;
        }

        t->vocab_size++;
        merges_done++;

        if (merges_done % 100 == 0 || merges_done == 1) {
            printf("  Merge %d: vocab_size=%d, pair=(%d,%d) count=%d\n",
                   merges_done, t->vocab_size, best_a, best_b, best_count);
            fflush(stdout);
        }
    }

    /* Compute max token length */
    t->max_token_len = 0;
    for (int i = 0; i < t->vocab_size; i++) {
        int len = strlen(t->tokens[i]);
        if (len > t->max_token_len) t->max_token_len = len;
    }

    printf("BPE complete: %d merges, vocab_size=%d, max_token_len=%d\n",
           merges_done, t->vocab_size, t->max_token_len);

    /* Cleanup */
    for (int r = 0; r < row_count; r++) {
        free(seqs[r]);
    }
    free(seqs);
    free(seqlens);
    free(pair_counts);
    free(dirty_rows);
    free(row_marked);
    free(seq);
}

/* ------------------------------------------------------------------ */
/*  write_tokenizer: serialize to a flat binary file                  */
/*                                                                    */
/*  Layout:  int32   block size (== MAX_TOKEN_LEN, for validation)    */
/*           int32   vocab_size                                       */
/*           vocab_size fixed MAX_TOKEN_LEN-byte blocks; block N is    */
/*           token ID N, NUL-terminated within the block.             */
/* ------------------------------------------------------------------ */
void write_tokenizer(const Tokenizer *t, const char *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror("fopen tokenizer");
        return;
    }

    int block_size = MAX_TOKEN_LEN;
    fwrite(&block_size, sizeof(int), 1, fp);
    fwrite(&t->vocab_size, sizeof(int), 1, fp);
    fwrite(t->tokens, MAX_TOKEN_LEN, t->vocab_size, fp);

    fclose(fp);
}

/* ------------------------------------------------------------------ */
/*  free_tokenizer: no dynamic memory to free (struct is stack-alloc) */
/* ------------------------------------------------------------------ */
void free_tokenizer(Tokenizer *t) {
    (void)t;
}

/* ------------------------------------------------------------------ */
/*  split_str: split a string by "<END>" delimiter                    */
/* ------------------------------------------------------------------ */
char **split_str(char *original, int *arr_length) {
    *arr_length = 1;
    char *cursor = original;
    while ((cursor = strstr(cursor, "<END>")) != NULL) {
        (*arr_length)++;
        cursor += 5;
    }

    char **output = malloc((*arr_length) * sizeof(char *));
    cursor = original;
    char *old_cursor = original;
    int i = 0;

    while ((cursor = strstr(cursor, "<END>")) != NULL) {
        int str_len = cursor - old_cursor;
        output[i] = malloc(str_len + 1);
        if (!output[i]) {
            fprintf(stderr, "malloc failed in split_str\n");
            for (int j = 0; j < i; j++) free(output[j]);
            free(output);
            return NULL;
        }
        strncpy(output[i], old_cursor, str_len);
        output[i][str_len] = '\0';
        cursor += 5;
        old_cursor = cursor;
        i++;
    }

    output[i] = malloc(strlen(old_cursor) + 1);
    if (!output[i]) {
        fprintf(stderr, "malloc failed in split_str (last element)\n");
        for (int j = 0; j < i; j++) free(output[j]);
        free(output);
        return NULL;
    }
    strcpy(output[i], old_cursor);

    return output;
}

/* ------------------------------------------------------------------ */
/*  free_rows: free all strings and the array returned by split_str   */
/* ------------------------------------------------------------------ */
void free_rows(char **rows, int count) {
    if (!rows) return;
    for (int i = 0; i < count; i++) {
        free(rows[i]);
    }
    free(rows);
}

/* ------------------------------------------------------------------ */
/*  read_file: read entire file into memory                           */
/* ------------------------------------------------------------------ */
char *read_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("fopen");
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        perror("fseek end");
        fclose(fp);
        return NULL;
    }

    long file_size = ftell(fp);
    if (file_size < 0) {
        perror("ftell");
        fclose(fp);
        return NULL;
    }

    if (fseek(fp, 0L, SEEK_SET) != 0) {
        perror("fseek start");
        fclose(fp);
        return NULL;
    }

    char *file_buffer = malloc(file_size + 1);
    if (!file_buffer) {
        perror("malloc file buffer");
        fclose(fp);
        return NULL;
    }

    if (fread(file_buffer, 1, file_size, fp) != (size_t)file_size) {
        perror("fread");
        free(file_buffer);
        fclose(fp);
        return NULL;
    }
    file_buffer[file_size] = '\0';
    fclose(fp);

    return file_buffer;
}

/* ------------------------------------------------------------------ */
/*  main                                                              */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]) {
    const char *input_path = "dataset/train.txt";
    const char *output_path = "tokenizer.bin";

    if (argc >= 2) input_path = argv[1];
    if (argc >= 3) output_path = argv[2];

    printf("Building tokenizer from %s ...\n", input_path);

    /* Read dataset */
    char *raw_text = read_file(input_path);
    if (!raw_text) {
        fprintf(stderr, "Failed to read %s\n", input_path);
        return 1;
    }
    printf("Read %zu bytes\n", strlen(raw_text));

    /* Split into stories for base vocab discovery */
    int dataset_length = 0;
    char **dataset_rows = split_str(raw_text, &dataset_length);
    printf("Loaded %d stories\n", dataset_length);

    /* Initialize tokenizer */
    Tokenizer tokenizer;
    init_tokenizer(&tokenizer);

    /* Build base vocabulary from characters */
    build_base_vocab(&tokenizer);

    /* Run BPE training */
    build_tokenizer(&tokenizer, dataset_rows, dataset_length);

    /* Write tokenizer to disk */
    write_tokenizer(&tokenizer, output_path);
    printf("Tokenizer written to %s\n", output_path);

    /* Cleanup */
    free_tokenizer(&tokenizer);
    free(raw_text);
    free_rows(dataset_rows, dataset_length);

    printf("Done.\n");
    return 0;
}
