#include "tokenizer.h"

/* Global trie storage — too large for the stack (64MB). Only the encoder
 * needs it, so it lives here rather than in the shared header. */
static TrieNode g_trie[MAX_TRIE_NODES];
static int g_trie_root;
static int g_trie_next;

/* ------------------------------------------------------------------ */
/*  trie_alloc: allocate next node from the pool                      */
/* ------------------------------------------------------------------ */
static int trie_alloc(void) {
    if (g_trie_next >= MAX_TRIE_NODES) return -1;
    g_trie[g_trie_next].token_id = -1;
    memset(g_trie[g_trie_next].children, -1, sizeof(g_trie[0].children));
    return g_trie_next++;
}

static char *read_file(const char *path);

/* ------------------------------------------------------------------ */
/*  load_tokenizer: read the flat binary tokenizer and build the trie */
/*  in a single pass. See write_tokenizer() for the file layout.      */
/* ------------------------------------------------------------------ */
int load_tokenizer(const char *path, Tokenizer *t) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("fopen tokenizer"); return -1; }

    int block_size = 0;
    if (fread(&block_size, sizeof(int), 1, fp) != 1 ||
        fread(&t->vocab_size, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Truncated tokenizer header\n");
        fclose(fp);
        return -1;
    }
    if (block_size != MAX_TOKEN_LEN) {
        fprintf(stderr, "Token block size mismatch: file=%d, build=%d\n",
                block_size, MAX_TOKEN_LEN);
        fclose(fp);
        return -1;
    }
    if (t->vocab_size < 0 || t->vocab_size > MAX_VOCAB) {
        fprintf(stderr, "Bad vocab_size %d\n", t->vocab_size);
        fclose(fp);
        return -1;
    }

    /* Initialise the trie pool with a root node. */
    g_trie_next = 0;
    g_trie_root = trie_alloc();
    t->trie = g_trie;
    t->trie_root = g_trie_root;

    /* Read each fixed-size block (block N = token ID N), inserting it into the
     * trie as we go and tracking the longest token (max_token_len isn't stored
     * in the file — it's cheap to recompute here). */
    t->max_token_len = 0;
    for (int id = 0; id < t->vocab_size; id++) {
        if (fread(t->tokens[id], MAX_TOKEN_LEN, 1, fp) != 1) {
            fprintf(stderr, "Truncated token data at id %d\n", id);
            fclose(fp);
            return -1;
        }
        t->tokens[id][MAX_TOKEN_LEN - 1] = '\0';  /* defensive */

        const char *tok = t->tokens[id];
        int node = t->trie_root;
        for (int c = 0; tok[c]; c++) {
            unsigned char ch = (unsigned char)tok[c];
            if (t->trie[node].children[ch] == -1) {
                int child = trie_alloc();
                if (child == -1) {
                    fprintf(stderr, "Trie pool exhausted at token %d\n", id);
                    fclose(fp);
                    return -1;
                }
                t->trie[node].children[ch] = child;
            }
            node = t->trie[node].children[ch];
        }
        t->trie[node].token_id = id;  /* mark end of token */

        int len = (int)strlen(tok);
        if (len > t->max_token_len) t->max_token_len = len;
    }

    fclose(fp);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  encode: text -> array of token IDs (BPE-aware greedy longest-match) */
/* ------------------------------------------------------------------ */
int *encode(const Tokenizer *t, const char *text, int *out_len) {
    /* First pass: count output length */
    int len = 0;
    const char *s = text;
    while (*s) {
        const char *best_end = NULL;

        int node = t->trie_root;
        const char *scan = s;

        while (*scan && node != -1) {
            unsigned char ch = (unsigned char)*scan;
            int child = t->trie[node].children[ch];
            if (child == -1) break;
            node = child;
            if (t->trie[node].token_id >= 0) {
                best_end = scan + 1;
            }
            scan++;
        }

        if (best_end) {
            len += (best_end - s);
            s = best_end;
        } else {
            len++;
            s++;
        }
    }

    /* Allocate output (+2 for BOS and EOS) */
    int *ids = malloc((len + 2) * sizeof(int));
    if (!ids) return NULL;

    /* Second pass: fill IDs */
    int pos = 0;
    ids[pos++] = TOK_BOS;
    s = text;

    while (*s) {
        const char *best_end = NULL;
        int best_id = -1;

        int node = t->trie_root;
        const char *scan = s;

        while (*scan && node != -1) {
            unsigned char ch = (unsigned char)*scan;
            int child = t->trie[node].children[ch];
            if (child == -1) break;
            node = child;
            if (t->trie[node].token_id >= 0) {
                best_end = scan + 1;
                best_id = t->trie[node].token_id;
            }
            scan++;
        }

        if (best_id >= 0) {
            ids[pos++] = best_id;
            s = best_end;
        } else {
            ids[pos++] = NUM_SPECIAL + (unsigned char)*s;
            s++;
        }
    }

    ids[pos++] = TOK_EOS;
    *out_len = pos;
    return ids;
}

/* ------------------------------------------------------------------ */
/*  decode: array of token IDs -> string                              */
/* ------------------------------------------------------------------ */
char *decode(const Tokenizer *t, const int *ids, int len, int *out_len) {
    /* Estimate max output size */
    char *out = malloc(len * MAX_TOKEN_LEN + 1);
    if (!out) return NULL;

    int pos = 0;
    for (int i = 0; i < len; i++) {
        if (ids[i] >= 0 && ids[i] < t->vocab_size) {
            const char *tok = t->tokens[ids[i]];
            while (*tok && pos < len * MAX_TOKEN_LEN) {
                out[pos++] = *tok++;
            }
        }
    }
    out[pos] = '\0';
    *out_len = pos;
    return out;
}

/* ------------------------------------------------------------------ */
/*  read_file: read entire file into memory (static to avoid conflict) */
/* ------------------------------------------------------------------ */
static char *read_file(const char *path) {
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
/*  print_usage: print CLI help                                       */
/* ------------------------------------------------------------------ */
static void print_usage(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s --tokenizer <path> --input <text|path> [--output <path>] [--decode]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --tokenizer PATH   Path to tokenizer.json (required)\n");
    fprintf(stderr, "  --input TEXT|PATH  Input text or file path (required)\n");
    fprintf(stderr, "  --output PATH      Write output to file instead of stdout\n");
    fprintf(stderr, "  --decode           Decode token IDs to text (default: encode)\n");
    fprintf(stderr, "  -h, --help         Show this help\n");
}

/* ------------------------------------------------------------------ */
/*  main — CLI for encoding / decoding                                */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]) {
    const char *tokenizer_path = NULL;
    const char *input_str = NULL;
    const char *input_path = NULL;
    const char *output_path = NULL;
    int decode_mode = 0;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--tokenizer") == 0 && ++i < argc) ||
            (strcmp(argv[i], "--input") == 0 && ++i < argc) ||
            (strcmp(argv[i], "--output") == 0 && ++i < argc)) {
            if (strcmp(argv[i - 1], "--tokenizer") == 0) tokenizer_path = argv[i];
            else if (strcmp(argv[i - 1], "--input") == 0) {
                /* Try as file path first, fall back to literal string */
                FILE *test = fopen(argv[i], "r");
                if (test) {
                    input_path = argv[i];
                    fclose(test);
                } else {
                    input_str = argv[i];
                }
            }
            else if (strcmp(argv[i - 1], "--output") == 0) output_path = argv[i];
        } else if (strcmp(argv[i], "--decode") == 0) {
            decode_mode = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!tokenizer_path || (!input_str && !input_path)) {
        fprintf(stderr, "Error: --tokenizer and --input are required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Load tokenizer (reads the binary file and builds the trie in one pass) */
    Tokenizer tokenizer;
    memset(&tokenizer, 0, sizeof(tokenizer));
    if (load_tokenizer(tokenizer_path, &tokenizer) != 0) {
        return 1;
    }

    /* Read input */
    char *input_text = NULL;
    if (input_str) {
        input_text = strdup(input_str);
    } else {
        input_text = read_file(input_path);
        if (!input_text) {
            fprintf(stderr, "Failed to read input file: %s\n", input_path);
            return 1;
        }
    }

    /* Strip trailing newline from file input */
    int input_len = strlen(input_text);
    while (input_len > 0 && (input_text[input_len - 1] == '\n' || input_text[input_len - 1] == '\r')) {
        input_text[--input_len] = '\0';
    }

    FILE *out_fp = stdout;
    if (output_path) {
        out_fp = fopen(output_path, "w");
        if (!out_fp) {
            perror("fopen output");
            free(input_text);
            return 1;
        }
    }

    if (!decode_mode) {
        /* Encode: text -> token IDs */
        int enc_len = 0;
        int *ids = encode(&tokenizer, input_text, &enc_len);
        if (!ids) {
            fprintf(stderr, "Encode failed\n");
            free(input_text);
            if (output_path) fclose(out_fp);
            return 1;
        }
        for (int i = 0; i < enc_len; i++) {
            fprintf(out_fp, "%d%s", ids[i], i < enc_len - 1 ? " " : "");
        }
        fprintf(out_fp, "\n");
        free(ids);
    } else {
        /* Decode: token IDs -> text */
        int *ids = malloc(MAX_VOCAB * sizeof(int));
        int id_count = 0;
        char *p = input_text;
        while (*p && id_count < MAX_VOCAB) {
            while (*p == ' ' || *p == ',') p++;
            if (*p == '\0') break;
            ids[id_count++] = atoi(p);
            while (*p && *p != ' ' && *p != ',') p++;
        }

        int dec_len = 0;
        char *decoded = decode(&tokenizer, ids, id_count, &dec_len);
        if (!decoded) {
            fprintf(stderr, "Decode failed\n");
            free(ids);
            free(input_text);
            if (output_path) fclose(out_fp);
            return 1;
        }
        fprintf(out_fp, "%s\n", decoded);
        free(decoded);
        free(ids);
    }

    if (output_path) fclose(out_fp);
    free(input_text);

    return 0;
}
