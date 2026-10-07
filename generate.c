/* generate.c — autoregressive sampling from a trained model. */

#include "tokenizer.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define EMB_DIM      384
#define N_LAYERS     8
#define N_HEADS      8
#define CONTEXT_LEN  256


static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --tokenizer tokenizer.bin --weights ckpt.bin --prompt \"...\" [options]\n"
        "Options:\n"
        "  --tokenizer PATH   (required)\n"
        "  --weights PATH     checkpoint with weights (required)\n"
        "  --prompt TEXT      (required)\n"
        "  --temperature F    sample (default: greedy)\n"
        "  --max_new_tokens N (default 64)\n"
        "  --seed N           (default 42)\n", prog);
}

int main(int argc, char **argv) {
    const char *tokenizer_path = "tokenizer.bin";
    const char *weights = NULL, *prompt = NULL;
    float temperature = -1.0f;
    int max_new_tokens = 64;
    unsigned int seed = 42;

    for (int i = 1; i < argc; i++) {
        char *k = argv[i], *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        #define NEEDV() do { if (!v) { usage(argv[0]); return 1; } i++; } while (0)
        if      (!strcmp(k, "--tokenizer"))      { NEEDV(); tokenizer_path = v; }
        else if (!strcmp(k, "--weights"))        { NEEDV(); weights = v; }
        else if (!strcmp(k, "--prompt"))         { NEEDV(); prompt = v; }
        else if (!strcmp(k, "--temperature"))    { NEEDV(); temperature = atof(v); }
        else if (!strcmp(k, "--max_new_tokens")) { NEEDV(); max_new_tokens = atoi(v); }
        else if (!strcmp(k, "--seed"))           { NEEDV(); seed = (unsigned int)strtoul(v, NULL, 10); }
        else { fprintf(stderr, "Unknown argument: %s\n", k); usage(argv[0]); return 1; }
        #undef NEEDV
    }
    if (!weights || !prompt) { usage(argv[0]); return 1; }
    srand(seed ? seed : 1);

    Tokenizer tok; memset(&tok, 0, sizeof(tok));
    if (load_tokenizer(tokenizer_path, &tok) != 0) return 1;

    Model m; memset(&m, 0, sizeof(m));
    if (model_load(&m, weights, NULL, NULL, NULL, NULL) != 0) { fprintf(stderr, "failed to load weights\n"); return 1; }

    /* encode prompt, drop trailing EOS so we continue mid-sequence */
    int elen = 0;
    int *enc = encode(&tok, prompt, &elen);
    int seq[CONTEXT_LEN];
    int L = elen - 1;               /* [BOS, prompt_tokens] */
    if (L <= 0) L = 1;              /* at least BOS */
    for (int i = 0; i < L; i++) seq[i] = enc[i];
    free(enc);

    int gen_count = 0;
    int *gen_ids = malloc(sizeof(int) * (size_t)max_new_tokens);
    float *logits = malloc(sizeof(float) * (size_t)CONTEXT_LEN * tok.vocab_size);

    for (int s = 0; s < max_new_tokens && L < CONTEXT_LEN; s++) {
        model_forward(&m, seq, 1, L, logits);
        float *row = logits + (size_t)(L - 1) * tok.vocab_size;
        int next;
        if (temperature > 0.0f) {
            float mx = row[0];
            for (int j = 1; j < tok.vocab_size; j++) if (row[j] > mx) mx = row[j];
            double ssum = 0;
            for (int j = 0; j < tok.vocab_size; j++) ssum += expf((row[j] - mx) / temperature);
            double u = ((double)rand() / (double)RAND_MAX) * ssum, acc = 0;
            next = tok.vocab_size - 1;
            for (int j = 0; j < tok.vocab_size; j++) { acc += expf((row[j] - mx) / temperature); if (acc >= u) { next = j; break; } }
        } else {
            int best = 0; float bv = row[0];
            for (int j = 1; j < tok.vocab_size; j++) if (row[j] > bv) { bv = row[j]; best = j; }
            next = best;
        }
        if (next == TOK_EOS) break;
        seq[L++] = next;
        if (next >= NUM_SPECIAL) gen_ids[gen_count++] = next;
    }

    /* decode generated content only (skip BOS/EOS/PAD) */
    int out_len = 0;
    char *text = decode(&tok, gen_ids, gen_count, &out_len);
    if (text) { fwrite(text, 1, out_len, stdout); fputc('\n', stdout); free(text); }

    free(gen_ids); free(logits);
    model_free(&m);
    return 0;
}
