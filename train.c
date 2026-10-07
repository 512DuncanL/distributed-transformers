/* train.c — train the transformer LM on dataset/train.txt.
 *
 * Pipeline: pre-tokenize the corpus into a flat uint16 buffer, bucket
 * stories by length, shuffle within buckets each epoch, and form batches
 * with a per-batch context length. Optimization: Adam (bias-corrected)
 * with warmup + cosine LR schedule and global-norm gradient clipping.
 */

#include "tokenizer.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* Fixed model configuration (matches tokenizer vocab 4096, PAD id 1). */
#define EMB_DIM      384
#define N_LAYERS     8
#define N_HEADS      8
#define CONTEXT_LEN  256

#define PI 3.14159265358979323846

/* ------------------------------------------------------------------ */
/*  Shuffle using stdlib rand()                                        */
/* ------------------------------------------------------------------ */
static void shuffle(int *a, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

/* ------------------------------------------------------------------ */
/*  Corpus                                                            */
/* ------------------------------------------------------------------ */
typedef struct {
    uint16_t *tokens;   /* flat uint16 buffer */
    size_t    n_tokens;
    uint32_t *offset;   /* per-story start into tokens */
    uint16_t *len;      /* per-story valid (non-PAD) length */
    int       n_stories;
} Corpus;

static char *read_entire_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen corpus"); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    buf[sz] = '\0';
    fclose(f);
    if (out_size) *out_size = (size_t)sz;
    return buf;
}

static int corpus_build(Corpus *c, const char *path, Tokenizer *tok, int max_stories) {
    memset(c, 0, sizeof(*c));
    size_t fsize = 0;
    char *buf = read_entire_file(path, &fsize);
    if (!buf) return -1;
    /* count stories + max story byte length */
    int n = 0; size_t maxlen = 0;
    char *s = buf, *end;
    while ((end = strstr(s, "<END>")) != NULL) {
        size_t len = (size_t)(end - s);
        if (len > maxlen) maxlen = len;
        n++; s = end + 5;
    }
    if (n == 0) { free(buf); return -1; }
    int n_target = (max_stories > 0 && max_stories < n) ? max_stories : n;

    char *sbuf = malloc(maxlen + 1);
    uint32_t *offs = malloc((size_t)n_target * sizeof(uint32_t));
    uint16_t *lens = malloc((size_t)n_target * sizeof(uint16_t));
    if (!sbuf || !offs || !lens) { free(buf); free(sbuf); free(offs); free(lens); return -1; }

    size_t cap = 1 << 20, total = 0;
    uint16_t *toks = malloc(cap * sizeof(uint16_t));
    if (!toks) { free(buf); free(sbuf); free(offs); free(lens); return -1; }

    s = buf; int idx = 0;
    while (idx < n_target && (end = strstr(s, "<END>")) != NULL) {
        size_t len = (size_t)(end - s);
        memcpy(sbuf, s, len); sbuf[len] = '\0';
        int elen = 0;
        int *ids = encode(tok, sbuf, &elen);
        int valid = (elen > CONTEXT_LEN) ? CONTEXT_LEN : elen;
        if (total + valid > cap) {
            while (total + valid > cap) cap <<= 1;
            uint16_t *nb = realloc(toks, cap * sizeof(uint16_t));
            if (!nb) { free(ids); free(toks); free(buf); free(sbuf); free(offs); free(lens); return -1; }
            toks = nb;
        }
        offs[idx] = (uint32_t)total;
        lens[idx] = (uint16_t)valid;
        if (elen > CONTEXT_LEN) {
            /* keep BOS + first 254 content + EOS */
            toks[total++] = (uint16_t)ids[0];
            for (int t = 1; t <= CONTEXT_LEN - 2; t++) toks[total++] = (uint16_t)ids[t];
            toks[total++] = (uint16_t)ids[elen - 1];
        } else {
            for (int t = 0; t < elen; t++) toks[total++] = (uint16_t)ids[t];
        }
        free(ids);
        s = end + 5; idx++;
    }
    free(buf); free(sbuf);
    c->tokens = toks; c->n_tokens = total;
    c->offset = offs; c->len = lens; c->n_stories = idx;
    printf("corpus: %d stories, %zu tokens (%.1fM)\n", idx, total, total / 1e6);
    return 0;
}

static void corpus_free(Corpus *c) {
    free(c->tokens); free(c->offset); free(c->len);
    memset(c, 0, sizeof(*c));
}

/* ------------------------------------------------------------------ */
/*  LR schedule                                                       */
/* ------------------------------------------------------------------ */
static float lr_schedule(int step, float peak, int warmup, int total) {
    if (step < warmup) return peak * (float)step / (float)warmup;
    float t = (float)(step - warmup) / (float)(total - warmup);
    if (t > 1.0f) t = 1.0f;
    return 0.1f * peak + 0.5f * (peak - 0.1f * peak) * (1.0f + cosf(PI * t));
}

/* ------------------------------------------------------------------ */
/*  CLI / usage                                                       */
/* ------------------------------------------------------------------ */
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --tokenizer tokenizer.bin [options]\n"
        "Options:\n"
        "  --tokenizer PATH    (required)\n"
        "  --corpus PATH       dataset/train.txt (default)\n"
        "  --resume PATH       resume from checkpoint (weights + Adam)\n"
        "  --weights PATH      init weights only (fresh Adam)\n"
        "  --out PATH          checkpoint output (default ckpt.bin)\n"
        "  --batch_size N      (default 32)\n"
        "  --lr F              peak learning rate (default 3e-4)\n"
        "  --epochs N          (default 6)\n"
        "  --warmup N          warmup steps (default 500)\n"
        "  --log_every N       log every N steps (default 100)\n"
        "  --ckpt_every N      checkpoint every N steps (default 5000)\n"
        "  --seed N            (default 42)\n"
        "  --max_stories N     limit corpus size (smoke tests)\n"
        "  --max_steps N       stop after N steps (smoke tests)\n",
        prog);
}

int main(int argc, char **argv) {
    const char *tokenizer_path = "tokenizer.bin";
    const char *corpus_path = "dataset/train.txt";
    const char *resume = NULL, *weights = NULL, *out = "ckpt.bin";
    int batch_size = 32, epochs = 6, warmup = 500;
    int log_every = 100, ckpt_every = 5000;
    float lr = 3e-4f;
    unsigned int seed = 42;
    long max_stories = -1, max_steps = -1;

    for (int i = 1; i < argc; i++) {
        char *k = argv[i], *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        #define NEEDV() do { if (!v) { usage(argv[0]); return 1; } i++; } while (0)
        if      (!strcmp(k, "--tokenizer"))  { NEEDV(); tokenizer_path = v; }
        else if (!strcmp(k, "--corpus"))     { NEEDV(); corpus_path = v; }
        else if (!strcmp(k, "--resume"))     { NEEDV(); resume = v; }
        else if (!strcmp(k, "--weights"))    { NEEDV(); weights = v; }
        else if (!strcmp(k, "--out"))        { NEEDV(); out = v; }
        else if (!strcmp(k, "--batch_size")) { NEEDV(); batch_size = atoi(v); }
        else if (!strcmp(k, "--lr"))         { NEEDV(); lr = atof(v); }
        else if (!strcmp(k, "--epochs"))     { NEEDV(); epochs = atoi(v); }
        else if (!strcmp(k, "--warmup"))     { NEEDV(); warmup = atoi(v); }
        else if (!strcmp(k, "--log_every"))  { NEEDV(); log_every = atoi(v); }
        else if (!strcmp(k, "--ckpt_every")) { NEEDV(); ckpt_every = atoi(v); }
        else if (!strcmp(k, "--seed"))       { NEEDV(); seed = (unsigned int)strtoul(v, NULL, 10); }
        else if (!strcmp(k, "--max_stories")){ NEEDV(); max_stories = atol(v); }
        else if (!strcmp(k, "--max_steps"))  { NEEDV(); max_steps = atol(v); }
        else { fprintf(stderr, "Unknown argument: %s\n", k); usage(argv[0]); return 1; }
        #undef NEEDV
    }

    srand(seed);

    /* tokenizer */
    Tokenizer tok; memset(&tok, 0, sizeof(tok));
    if (load_tokenizer(tokenizer_path, &tok) != 0) return 1;

    /* corpus */
    Corpus corpus;
    if (corpus_build(&corpus, corpus_path, &tok, (int)max_stories) != 0) { fprintf(stderr, "corpus build failed\n"); return 1; }

    /* model config */
    ModelConfig cfg;
    cfg.vocab_size = tok.vocab_size;
    cfg.context_len = CONTEXT_LEN;
    cfg.emb_dim = EMB_DIM;
    cfg.n_layers = N_LAYERS;
    cfg.n_heads = N_HEADS;
    cfg.pad_token_id = TOK_PAD;

    Model m; memset(&m, 0, sizeof(m));
    float *m_opt = NULL, *v_opt = NULL;
    int has_optim = 0, start_step = 0;
    if (resume) {
        if (model_load(&m, resume, &m_opt, &v_opt, &has_optim, &start_step) != 0) { fprintf(stderr, "resume load failed\n"); return 1; }
        if (!has_optim) {
            m_opt = calloc(m.n_params, sizeof(float));
            v_opt = calloc(m.n_params, sizeof(float));
        }
        printf("resumed from %s (step %d)\n", resume, start_step);
    } else if (weights) {
        if (model_load(&m, weights, NULL, NULL, NULL, NULL) != 0) { fprintf(stderr, "weights load failed\n"); return 1; }
        m_opt = calloc(m.n_params, sizeof(float));
        v_opt = calloc(m.n_params, sizeof(float));
        printf("initialized from weights %s\n", weights);
    } else {
        if (model_init(&m, &cfg, seed) != 0) { fprintf(stderr, "model init failed\n"); return 1; }
        m_opt = calloc(m.n_params, sizeof(float));
        v_opt = calloc(m.n_params, sizeof(float));
        printf("fresh init\n");
    }
    if (!m_opt || !v_opt) { fprintf(stderr, "optimizer alloc failed\n"); return 1; }
    printf("model: vocab=%d D=%d layers=%d heads=%d params=%zu (%.1fM)\n",
           cfg.vocab_size, cfg.emb_dim, cfg.n_layers, cfg.n_heads, m.n_params, m.n_params / 1e6);

    /* length buckets */
    int *bucket_id = malloc((size_t)corpus.n_stories * sizeof(int));
    int bucket_count[16] = {0};
    for (int i = 0; i < corpus.n_stories; i++) {
        int l = corpus.len[i];
        int b = (l >= CONTEXT_LEN) ? 15 : (l / 16);
        bucket_id[i] = b; bucket_count[b]++;
    }
    int *bucket[16];
    for (int b = 0; b < 16; b++) bucket[b] = malloc((size_t)(bucket_count[b] ? bucket_count[b] : 1) * sizeof(int));
    int fill[16] = {0};
    for (int i = 0; i < corpus.n_stories; i++) bucket[bucket_id[i]][fill[bucket_id[i]]++] = i;

    int *ids = malloc((size_t)batch_size * CONTEXT_LEN * sizeof(int));
    if (!ids) { fprintf(stderr, "ids alloc failed\n"); return 1; }

    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, grad_clip = 1.0f;
    int total_steps = start_step + (int)((long)epochs * (corpus.n_stories / batch_size + 1));
    int step = start_step;
    int bucket_order[16];
    for (int b = 0; b < 16; b++) bucket_order[b] = b;

    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    long long tokens_done = 0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle(bucket_order, 16);
        for (int bi = 0; bi < 16; bi++) {
            int b = bucket_order[bi];
            if (bucket_count[b] == 0) continue;
            shuffle(bucket[b], bucket_count[b]);
            for (int start = 0; start < bucket_count[b]; start += batch_size) {
                int nb = bucket_count[b] - start; if (nb > batch_size) nb = batch_size;
                int L_b = 0;
                for (int j = 0; j < nb; j++) { int l = corpus.len[bucket[b][start + j]]; if (l > L_b) L_b = l; }
                L_b = ((L_b + 7) / 8) * 8;
                for (int j = 0; j < nb; j++) {
                    int si = bucket[b][start + j];
                    const uint16_t *src = corpus.tokens + corpus.offset[si];
                    int *dst = ids + (size_t)j * L_b;
                    int l = corpus.len[si];
                    for (int t = 0; t < l; t++) dst[t] = (int)src[t];
                    for (int t = l; t < L_b; t++) dst[t] = TOK_PAD;
                }
                float loss = model_train_step(&m, ids, nb, L_b, NULL, 1);
                float lr_cur = lr_schedule(step, lr, warmup, total_steps);
                adam_step(&m, m_opt, v_opt, lr_cur, beta1, beta2, eps, step + 1, grad_clip);
                tokens_done += (long long)nb * L_b;
                step++;
                if (step % log_every == 0) {
                    clock_gettime(CLOCK_MONOTONIC, &t_now);
                    double dt = (t_now.tv_sec - t_start.tv_sec) + (t_now.tv_nsec - t_start.tv_nsec) / 1e9;
                    printf("step %d epoch %d loss %.4f lr %.2e tok/s %.0f elapsed %.0fs\n",
                           step, epoch, loss, lr_cur, tokens_done / (dt > 0 ? dt : 1e-9), dt);
                    fflush(stdout);
                }
                if (step % ckpt_every == 0) {
                    if (model_save(&m, out, m_opt, v_opt, step) != 0) fprintf(stderr, "checkpoint save failed\n");
                    else printf("checkpoint saved: %s (step %d)\n", out, step);
                }
                if (max_steps > 0 && step >= max_steps) break;
            }
            if (max_steps > 0 && step >= max_steps) break;
        }
        if (model_save(&m, out, m_opt, v_opt, step) != 0) fprintf(stderr, "epoch checkpoint save failed\n");
        else printf("epoch %d done — checkpoint saved: %s\n", epoch, out);
        if (max_steps > 0 && step >= max_steps) break;
    }

    model_save(&m, out, m_opt, v_opt, step);
    printf("training complete: %d steps, final checkpoint %s\n", step, out);

    free(ids);
    free(bucket_id);
    for (int b = 0; b < 16; b++) free(bucket[b]);
    free(m_opt); free(v_opt);
    model_free(&m);
    corpus_free(&corpus);
    return 0;
}
