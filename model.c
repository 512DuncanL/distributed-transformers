/* model.c — decoder-only transformer LM: forward, backward, init, save/load,
 * Adam. All GEMMs go through OpenBLAS cblas_sgemm (row-major). */

#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cblas.h>

#define PI 3.14159265358979323846

/* ------------------------------------------------------------------ */
/*  Thin GEMM wrapper (row-major). transA/transB: 0=NoTrans, 1=Trans.  */
/* ------------------------------------------------------------------ */
static inline void gemm(int transA, int transB, int M, int N, int K,
                        float alpha, const float *A, int lda,
                        const float *B, int ldb, float beta, float *C, int ldc) {
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

/* ------------------------------------------------------------------ */
/*  RNG (splitmix64 + Box-Muller) for deterministic init               */
/* ------------------------------------------------------------------ */
static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double rand_uniform(uint64_t *s) {
    return (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
}
static float rand_normal(uint64_t *s) {
    double u1 = rand_uniform(s), u2 = rand_uniform(s);
    if (u1 <= 0.0) u1 = 1e-12;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2));
}
static void fill_normal(float *x, size_t n, uint64_t *s, float std) {
    for (size_t i = 0; i < n; i++) x[i] = rand_normal(s) * std;
}

/* ------------------------------------------------------------------ */
/*  Layer parameter pointers (params + matching grads)                 */
/* ------------------------------------------------------------------ */
typedef struct {
    float *n1g, *WQ, *WK, *WV, *WO, *n2g, *Wg, *Wu, *Wd;
    float *dn1g,*dWQ,*dWK,*dWV,*dWO,*dn2g,*dWg,*dWu,*dWd;
} LayerPtrs;

/* forward activations cached for the backward pass (one per layer) */
typedef struct {
    float *x_in, *x_mid, *h1, *q, *k, *v, *attn, *h2;
    float *gate, *act, *up;
    float *inv1, *inv2;
    float *probs;   /* [B*H*L*L] */
} LayerCache;

typedef struct {
    int B, L, H;
    LayerCache *lc;      /* n_layers entries */
    float *buf; size_t cap;
    /* single buffers */
    float *x, *xf, *invf, *logits;
    float *dx, *dx_mid, *dh, *dattn, *dq, *dk, *dv;
    float *h4a, *h4b, *h4c;
    float *dp;           /* [L*L] per-head scratch */
} Workspace;

typedef struct {
    LayerPtrs *L;        /* n_layers entries */
    float *tok_emb, *pos_emb, *final_gamma, *lm_head;
    float *dtok_emb, *dpos_emb, *dfinal_gamma, *dlm_head;
    Workspace ws;
} Priv;

/* ------------------------------------------------------------------ */
/*  Geometry / binding / init                                          */
/* ------------------------------------------------------------------ */
int model_compute_geometry(const ModelConfig *cfg, int *head_dim,
                           int *mlp_hidden, size_t *n_params) {
    if (cfg->emb_dim <= 0 || cfg->n_heads <= 0 || cfg->emb_dim % cfg->n_heads != 0)
        return -1;
    int D = cfg->emb_dim, H = cfg->n_heads, d = D / H, H4 = 4 * D;
    int V = cfg->vocab_size, L = cfg->context_len, nl = cfg->n_layers;
    if (head_dim)   *head_dim   = d;
    if (mlp_hidden) *mlp_hidden = H4;
    size_t n = 0;
    n += (size_t)V * D;                       /* tok_emb */
    n += (size_t)L * D;                       /* pos_emb */
    for (int l = 0; l < nl; l++) {
        n += (size_t)D;                       /* n1g */
        n += (size_t)D * D;                   /* WQ */
        n += (size_t)D * D;                   /* WK */
        n += (size_t)D * D;                   /* WV */
        n += (size_t)D * D;                   /* WO */
        n += (size_t)D;                       /* n2g */
        n += (size_t)H4 * D;                  /* Wg */
        n += (size_t)H4 * D;                  /* Wu */
        n += (size_t)D * H4;                  /* Wd */
    }
    n += (size_t)D;                           /* final gamma */
    n += (size_t)V * D;                       /* lm_head */
    if (n_params) *n_params = n;
    return 0;
}

static void bind(Model *m) {
    Priv *p = m->priv;
    const ModelConfig *c = &m->cfg;
    int D = c->emb_dim, H4 = m->mlp_hidden, V = c->vocab_size, L = c->context_len;
    float *w = m->params, *g = m->grads;
    size_t o = 0;
    p->tok_emb = w + o; p->dtok_emb = g + o; o += (size_t)V * D;
    p->pos_emb = w + o; p->dpos_emb = g + o; o += (size_t)L * D;
    for (int l = 0; l < c->n_layers; l++) {
        LayerPtrs *lp = &p->L[l];
        lp->n1g = w+o; lp->dn1g = g+o; o += D;
        lp->WQ  = w+o; lp->dWQ  = g+o; o += (size_t)D*D;
        lp->WK  = w+o; lp->dWK  = g+o; o += (size_t)D*D;
        lp->WV  = w+o; lp->dWV  = g+o; o += (size_t)D*D;
        lp->WO  = w+o; lp->dWO  = g+o; o += (size_t)D*D;
        lp->n2g = w+o; lp->dn2g = g+o; o += D;
        lp->Wg  = w+o; lp->dWg  = g+o; o += (size_t)H4*D;
        lp->Wu  = w+o; lp->dWu  = g+o; o += (size_t)H4*D;
        lp->Wd  = w+o; lp->dWd  = g+o; o += (size_t)D*H4;
    }
    p->final_gamma = w+o; p->dfinal_gamma = g+o; o += D;
    p->lm_head     = w+o; p->dlm_head     = g+o; o += (size_t)V*D;
}

static void init_weights(Model *m, unsigned int seed) {
    Priv *p = m->priv;
    const ModelConfig *c = &m->cfg;
    int D = c->emb_dim, H4 = m->mlp_hidden, V = c->vocab_size, L = c->context_len;
    uint64_t s = seed ? seed : 1;
    fill_normal(p->tok_emb, (size_t)V * D, &s, 0.02f);
    fill_normal(p->pos_emb, (size_t)L * D, &s, 0.01f);
    float res = 1.0f / sqrtf(2.0f * (float)c->n_layers);
    for (int l = 0; l < c->n_layers; l++) {
        LayerPtrs *lp = &p->L[l];
        for (int j = 0; j < D; j++) { lp->n1g[j] = 1.0f; lp->n2g[j] = 1.0f; }
        fill_normal(lp->WQ, (size_t)D * D, &s, 0.02f);
        fill_normal(lp->WK, (size_t)D * D, &s, 0.02f);
        fill_normal(lp->WV, (size_t)D * D, &s, 0.02f);
        fill_normal(lp->WO, (size_t)D * D, &s, 0.02f * res);
        fill_normal(lp->Wg, (size_t)H4 * D, &s, 0.02f);
        fill_normal(lp->Wu, (size_t)H4 * D, &s, 0.02f);
        fill_normal(lp->Wd, (size_t)D * H4, &s, 0.02f * res);
    }
    for (int j = 0; j < D; j++) p->final_gamma[j] = 1.0f;
    fill_normal(p->lm_head, (size_t)V * D, &s, 0.02f);
}

int model_init(Model *m, const ModelConfig *cfg, unsigned int seed) {
    int hd, h4; size_t n;
    if (model_compute_geometry(cfg, &hd, &h4, &n) != 0) return -1;
    memset(m, 0, sizeof(*m));
    m->cfg = *cfg; m->head_dim = hd; m->mlp_hidden = h4; m->n_params = n;
    m->params = calloc(n, sizeof(float));
    m->grads  = calloc(n, sizeof(float));
    if (!m->params || !m->grads) { model_free(m); return -1; }
    Priv *p = calloc(1, sizeof(Priv));
    if (!p) { model_free(m); return -1; }
    p->L = calloc(cfg->n_layers, sizeof(LayerPtrs));
    p->ws.lc = calloc(cfg->n_layers, sizeof(LayerCache));
    if (!p->L || !p->ws.lc) { model_free(m); free(p->L); free(p->ws.lc); free(p); return -1; }
    m->priv = p;
    bind(m);
    init_weights(m, seed);
    return 0;
}

void model_free(Model *m) {
    if (!m) return;
    Priv *p = m->priv;
    if (p) { free(p->ws.buf); free(p->ws.lc); free(p->L); free(p); }
    free(m->params);
    free(m->grads);
    memset(m, 0, sizeof(*m));
}

/* ------------------------------------------------------------------ */
/*  Workspace allocation                                               */
/* ------------------------------------------------------------------ */
static float *carve(float **cursor, size_t n) { float *r = *cursor; *cursor += n; return r; }

static int ws_ensure(Priv *p, const ModelConfig *c, int B, int L) {
    int D = c->emb_dim, H = c->n_heads, H4 = 4 * D, V = c->vocab_size, nl = c->n_layers;
    size_t BL = (size_t)B * L;
    size_t per_layer = BL * D * 8 + BL * H4 * 3 + BL * 2 + (size_t)B * H * L * L;
    size_t total = per_layer * nl
        + BL * D * 2   /* x, xf */
        + BL           /* invf */
        + BL * V       /* logits */
        + BL * D * 7   /* dx,dx_mid,dh,dattn,dq,dk,dv */
        + BL * H4 * 3  /* h4a,h4b,h4c */
        + (size_t)L * L; /* dp */
    if (total * sizeof(float) > p->ws.cap) {
        float *nb = realloc(p->ws.buf, total * sizeof(float));
        if (!nb) return -1;
        p->ws.buf = nb; p->ws.cap = total * sizeof(float);
    }
    float *cur = p->ws.buf;
    p->ws.B = B; p->ws.L = L; p->ws.H = H;
    for (int l = 0; l < nl; l++) {
        LayerCache *lc = &p->ws.lc[l];
        lc->x_in  = carve(&cur, BL*D);
        lc->x_mid = carve(&cur, BL*D);
        lc->h1    = carve(&cur, BL*D);
        lc->q     = carve(&cur, BL*D);
        lc->k     = carve(&cur, BL*D);
        lc->v     = carve(&cur, BL*D);
        lc->attn  = carve(&cur, BL*D);
        lc->h2    = carve(&cur, BL*D);
        lc->gate  = carve(&cur, BL*H4);
        lc->act   = carve(&cur, BL*H4);
        lc->up    = carve(&cur, BL*H4);
        lc->inv1  = carve(&cur, BL);
        lc->inv2  = carve(&cur, BL);
        lc->probs = carve(&cur, (size_t)B*H*L*L);
    }
    p->ws.x      = carve(&cur, BL*D);
    p->ws.xf     = carve(&cur, BL*D);
    p->ws.invf   = carve(&cur, BL);
    p->ws.logits = carve(&cur, BL*V);
    p->ws.dx     = carve(&cur, BL*D);
    p->ws.dx_mid = carve(&cur, BL*D);
    p->ws.dh     = carve(&cur, BL*D);
    p->ws.dattn  = carve(&cur, BL*D);
    p->ws.dq     = carve(&cur, BL*D);
    p->ws.dk     = carve(&cur, BL*D);
    p->ws.dv     = carve(&cur, BL*D);
    p->ws.h4a    = carve(&cur, BL*H4);
    p->ws.h4b    = carve(&cur, BL*H4);
    p->ws.h4c    = carve(&cur, BL*H4);
    p->ws.dp     = carve(&cur, (size_t)L*L);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Elementwise primitives                                             */
/* ------------------------------------------------------------------ */
static void rmsnorm_fwd(float *y, float *inv, const float *x, int D, int N, const float *gamma) {
    const float eps = 1e-5f;
    for (int i = 0; i < N; i++) {
        const float *xr = x + (size_t)i * D;
        float *yr = y + (size_t)i * D;
        float ss = 0;
        for (int j = 0; j < D; j++) ss += xr[j] * xr[j];
        float r = sqrtf(ss / D + eps);
        float iv = 1.0f / r;
        inv[i] = iv;
        for (int j = 0; j < D; j++) yr[j] = xr[j] * iv * gamma[j];
    }
}

/* dy: grad wrt y; x: norm input; inv: stored 1/rms; gamma: norm weights;
 * dgamma: accumulator for gamma grads; dx may alias dy. */
static void rmsnorm_bwd(float *dy, const float *x, const float *inv, const float *gamma,
                        float *dx, float *dgamma, int D, int N) {
    float invD = 1.0f / (float)D;
    for (int i = 0; i < N; i++) {
        const float *xr = x + (size_t)i * D;
        float *dyr = dy + (size_t)i * D;
        float *dxr = dx + (size_t)i * D;
        float iv = inv[i];
        float S = 0;
        for (int j = 0; j < D; j++) S += xr[j] * dyr[j] * gamma[j];
        for (int j = 0; j < D; j++) dgamma[j] += dyr[j] * xr[j] * iv;
        float cc = iv * iv * iv * invD * S;
        for (int j = 0; j < D; j++) dxr[j] = iv * dyr[j] * gamma[j] - cc * xr[j];
    }
}

static void silu_fwd(float *y, const float *x, int N) {
    for (int i = 0; i < N; i++) {
        float e = x[i];
        float sg = 1.0f / (1.0f + expf(-e));
        y[i] = e * sg;
    }
}

/* dgate_out = dact_in * silu'(gate); in-place is allowed. */
static void silu_bwd(float *dgate_out, const float *gate, const float *dact_in, int N) {
    for (int i = 0; i < N; i++) {
        float x = gate[i];
        float sg = 1.0f / (1.0f + expf(-x));
        float g = sg * (1.0f + x * (1.0f - sg));
        dgate_out[i] = dact_in[i] * g;
    }
}

/* ------------------------------------------------------------------ */
/*  Attention (forward)                                                */
/* ------------------------------------------------------------------ */
static void attention_fwd(float *attn, float *probs, const float *q, const float *k,
                          const float *v, int B, int L, int H, int d) {
    int D = H * d;
    float scale = 1.0f / sqrtf((float)d);
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const float *qh = q + (size_t)b * L * D + h * d;
            const float *kh = k + (size_t)b * L * D + h * d;
            const float *vh = v + (size_t)b * L * D + h * d;
            float *sh = probs + (size_t)(b * H + h) * L * L;
            float *oh = attn + (size_t)b * L * D + h * d;
            gemm(0, 1, L, L, d, scale, qh, D, kh, D, 0.0f, sh, L);
            /* causal softmax, in place */
            for (int i = 0; i < L; i++) {
                float mx = -INFINITY;
                for (int j = 0; j <= i; j++) if (sh[i * L + j] > mx) mx = sh[i * L + j];
                float s = 0;
                for (int j = 0; j <= i; j++) { sh[i * L + j] = expf(sh[i * L + j] - mx); s += sh[i * L + j]; }
                float iv = 1.0f / s;
                for (int j = 0; j <= i; j++) sh[i * L + j] *= iv;
                for (int j = i + 1; j < L; j++) sh[i * L + j] = 0.0f;
            }
            gemm(0, 0, L, d, L, 1.0f, sh, L, vh, D, 0.0f, oh, d);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Forward core                                                       */
/* ------------------------------------------------------------------ */
static void forward_core(const Model *m, const int *ids, int B, int L,
                         float *logits_out, Workspace *ws) {
    Priv *p = m->priv;
    const ModelConfig *c = &m->cfg;
    int D = c->emb_dim, H = c->n_heads, d = m->head_dim, H4 = m->mlp_hidden, V = c->vocab_size;
    int BL = B * L;

    for (int i = 0; i < BL; i++) {
        int tok = ids[i], pos = i % L;
        const float *te = p->tok_emb + (size_t)tok * D;
        const float *pe = p->pos_emb + (size_t)pos * D;
        float *xr = ws->x + (size_t)i * D;
        for (int j = 0; j < D; j++) xr[j] = te[j] + pe[j];
    }

    float *x = ws->x;
    for (int l = 0; l < c->n_layers; l++) {
        LayerCache *lc = &ws->lc[l];
        LayerPtrs *lp = &p->L[l];
        memcpy(lc->x_in, x, sizeof(float) * BL * D);
        rmsnorm_fwd(lc->h1, lc->inv1, x, D, BL, lp->n1g);
        gemm(0, 1, BL, D, D, 1.0f, lc->h1, D, lp->WQ, D, 0.0f, lc->q, D);
        gemm(0, 1, BL, D, D, 1.0f, lc->h1, D, lp->WK, D, 0.0f, lc->k, D);
        gemm(0, 1, BL, D, D, 1.0f, lc->h1, D, lp->WV, D, 0.0f, lc->v, D);
        attention_fwd(lc->attn, lc->probs, lc->q, lc->k, lc->v, B, L, H, d);
        gemm(0, 1, BL, D, D, 1.0f, lc->attn, D, lp->WO, D, 0.0f, x, D);
        for (int i = 0; i < BL * D; i++) x[i] += lc->x_in[i];
        memcpy(lc->x_mid, x, sizeof(float) * BL * D);
        rmsnorm_fwd(lc->h2, lc->inv2, x, D, BL, lp->n2g);
        gemm(0, 1, BL, H4, D, 1.0f, lc->h2, D, lp->Wg, D, 0.0f, lc->gate, H4);
        gemm(0, 1, BL, H4, D, 1.0f, lc->h2, D, lp->Wu, D, 0.0f, lc->up, H4);
        silu_fwd(lc->act, lc->gate, BL * H4);
        float *a = ws->h4b;
        for (int i = 0; i < BL * H4; i++) a[i] = lc->act[i] * lc->up[i];
        gemm(0, 1, BL, D, H4, 1.0f, a, H4, lp->Wd, H4, 0.0f, x, D);
        for (int i = 0; i < BL * D; i++) x[i] += lc->x_mid[i];
    }

    rmsnorm_fwd(ws->xf, ws->invf, x, D, BL, p->final_gamma);
    gemm(0, 1, BL, V, D, 1.0f, ws->xf, D, p->lm_head, D, 0.0f, ws->logits, V);
    if (logits_out) memcpy(logits_out, ws->logits, sizeof(float) * BL * V);
}

/* ------------------------------------------------------------------ */
/*  Loss + dlogits (overwrites logits in place with dlogits)           */
/* ------------------------------------------------------------------ */
static float loss_and_dlogits(const int *ids, int B, int L, int vocab, int pad_id,
                              float *logits) {
    int BL = B * L;
    float loss = 0.0f; int cnt = 0;
    for (int i = 0; i < BL; i++) {
        int l = i % L;
        if (l == L - 1) continue;
        int tgt = ids[i + 1];
        if (tgt == pad_id) continue;
        float *row = logits + (size_t)i * vocab;
        float mx = row[0];
        for (int j = 1; j < vocab; j++) if (row[j] > mx) mx = row[j];
        float s = 0;
        for (int j = 0; j < vocab; j++) s += expf(row[j] - mx);
        loss += mx + logf(s) - row[tgt];
        cnt++;
    }
    if (cnt == 0) return 0.0f;
    loss /= (float)cnt;
    for (int i = 0; i < BL; i++) {
        int l = i % L;
        float *row = logits + (size_t)i * vocab;
        if (l == L - 1 || ids[i + 1] == pad_id) { memset(row, 0, sizeof(float) * vocab); continue; }
        int tgt = ids[i + 1];
        float mx = row[0];
        for (int j = 1; j < vocab; j++) if (row[j] > mx) mx = row[j];
        float s = 0;
        for (int j = 0; j < vocab; j++) s += expf(row[j] - mx);
        float iv = 1.0f / s, ic = 1.0f / (float)cnt;
        for (int j = 0; j < vocab; j++)
            row[j] = (expf(row[j] - mx) * iv - (j == tgt ? 1.0f : 0.0f)) * ic;
    }
    return loss;
}

/* ------------------------------------------------------------------ */
/*  Backward                                                           */
/* ------------------------------------------------------------------ */
static void backward_core(Model *m, const int *ids, int B, int L, Workspace *ws) {
    Priv *p = m->priv;
    const ModelConfig *c = &m->cfg;
    int D = c->emb_dim, H = c->n_heads, d = m->head_dim, H4 = m->mlp_hidden, V = c->vocab_size;
    int BL = B * L;
    float scale = 1.0f / sqrtf((float)d);

    /* lm_head + final norm: ws->logits currently holds dlogits. */
    gemm(0, 0, BL, D, V, 1.0f, ws->logits, V, p->lm_head, D, 0.0f, ws->dx, D);
    gemm(1, 0, V, D, BL, 1.0f, ws->logits, V, ws->xf, D, 1.0f, p->dlm_head, D);
    rmsnorm_bwd(ws->dx, ws->x, ws->invf, p->final_gamma, ws->dx, p->dfinal_gamma, D, BL);

    for (int l = c->n_layers - 1; l >= 0; l--) {
        LayerCache *lc = &ws->lc[l];
        LayerPtrs *lp = &p->L[l];

        /* ---- MLP sublayer (x_next = x_mid + mlp) ---- */
        memcpy(ws->dx_mid, ws->dx, sizeof(float) * BL * D);
        gemm(0, 0, BL, H4, D, 1.0f, ws->dx, D, lp->Wd, H4, 0.0f, ws->h4a, H4);
        for (int i = 0; i < BL * H4; i++) ws->h4b[i] = lc->act[i] * lc->up[i];   /* a */
        gemm(1, 0, D, H4, BL, 1.0f, ws->dx, D, ws->h4b, H4, 1.0f, lp->dWd, H4);
        for (int i = 0; i < BL * H4; i++) {
            float da = ws->h4a[i];
            ws->h4b[i] = da * lc->act[i];   /* dup */
            ws->h4c[i] = da * lc->up[i];    /* dact */
        }
        silu_bwd(ws->h4c, lc->gate, ws->h4c, BL * H4);   /* dgate in place */
        gemm(1, 0, H4, D, BL, 1.0f, ws->h4b, H4, lc->h2, D, 1.0f, lp->dWu, D);
        gemm(1, 0, H4, D, BL, 1.0f, ws->h4c, H4, lc->h2, D, 1.0f, lp->dWg, D);
        gemm(0, 0, BL, D, H4, 1.0f, ws->h4c, H4, lp->Wg, D, 0.0f, ws->dh, D);
        gemm(0, 0, BL, D, H4, 1.0f, ws->h4b, H4, lp->Wu, D, 1.0f, ws->dh, D);
        rmsnorm_bwd(ws->dh, lc->x_mid, lc->inv2, lp->n2g, ws->dx_mid, lp->dn2g, D, BL);

        /* ---- attention sublayer (x_mid = x_in + attn_out) ---- */
        gemm(1, 0, D, D, BL, 1.0f, ws->dx_mid, D, lc->attn, D, 1.0f, lp->dWO, D);
        gemm(0, 0, BL, D, D, 1.0f, ws->dx_mid, D, lp->WO, D, 0.0f, ws->dattn, D);
        memset(ws->dq, 0, sizeof(float) * BL * D);
        memset(ws->dk, 0, sizeof(float) * BL * D);
        memset(ws->dv, 0, sizeof(float) * BL * D);
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                const float *ph = lc->probs + (size_t)(b * H + h) * L * L;
                const float *qh = lc->q + (size_t)b * L * D + h * d;
                const float *kh = lc->k + (size_t)b * L * D + h * d;
                const float *vh = lc->v + (size_t)b * L * D + h * d;
                const float *doh = ws->dattn + (size_t)b * L * D + h * d;
                float *dqh = ws->dq + (size_t)b * L * D + h * d;
                float *dkh = ws->dk + (size_t)b * L * D + h * d;
                float *dvh = ws->dv + (size_t)b * L * D + h * d;
                float *dp = ws->dp;
                gemm(1, 0, L, d, L, 1.0f, ph, L, doh, D, 0.0f, dvh, D);   /* dv = P^T dout */
                gemm(0, 1, L, L, d, 1.0f, doh, D, vh, D, 0.0f, dp, L);   /* dp = dout V^T */
                for (int i = 0; i < L; i++) {
                    float r = 0;
                    for (int j = 0; j < L; j++) r += ph[i * L + j] * dp[i * L + j];
                    for (int j = 0; j < L; j++) dp[i * L + j] = ph[i * L + j] * (dp[i * L + j] - r);
                }
                gemm(0, 0, L, d, L, scale, dp, L, kh, D, 0.0f, dqh, D);  /* dq = ds K */
                gemm(1, 0, L, d, L, scale, dp, L, qh, D, 0.0f, dkh, D);  /* dk = ds^T Q */
            }
        }
        gemm(0, 0, BL, D, D, 1.0f, ws->dq, D, lp->WQ, D, 0.0f, ws->dh, D);
        gemm(0, 0, BL, D, D, 1.0f, ws->dk, D, lp->WK, D, 1.0f, ws->dh, D);
        gemm(0, 0, BL, D, D, 1.0f, ws->dv, D, lp->WV, D, 1.0f, ws->dh, D);
        gemm(1, 0, D, D, BL, 1.0f, ws->dq, D, lc->h1, D, 1.0f, lp->dWQ, D);
        gemm(1, 0, D, D, BL, 1.0f, ws->dk, D, lc->h1, D, 1.0f, lp->dWK, D);
        gemm(1, 0, D, D, BL, 1.0f, ws->dv, D, lc->h1, D, 1.0f, lp->dWV, D);
        memcpy(ws->dx, ws->dx_mid, sizeof(float) * BL * D);
        rmsnorm_bwd(ws->dh, lc->x_in, lc->inv1, lp->n1g, ws->dx, lp->dn1g, D, BL);
    }

    /* embedding gradients */
    for (int i = 0; i < BL; i++) {
        int tok = ids[i], pos = i % L;
        float *dxr = ws->dx + (size_t)i * D;
        float *dte = p->dtok_emb + (size_t)tok * D;
        float *dpe = p->dpos_emb + (size_t)pos * D;
        for (int j = 0; j < D; j++) { dte[j] += dxr[j]; dpe[j] += dxr[j]; }
    }
}

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */
void model_forward(const Model *m, const int *ids, int B, int L, float *logits) {
    Priv *p = m->priv;
    ws_ensure(p, &m->cfg, B, L);
    forward_core(m, ids, B, L, logits, &p->ws);
}

float model_train_step(Model *m, const int *ids, int B, int L,
                       float *logits, int compute_grad) {
    Priv *p = m->priv;
    ws_ensure(p, &m->cfg, B, L);
    forward_core(m, ids, B, L, logits, &p->ws);
    float loss = loss_and_dlogits(ids, B, L, m->cfg.vocab_size, m->cfg.pad_token_id, p->ws.logits);
    if (compute_grad) {
        memset(m->grads, 0, m->n_params * sizeof(float));
        backward_core(m, ids, B, L, &p->ws);
    }
    return loss;
}

int model_save(const Model *m, const char *path, const float *om, const float *ov, int step) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int32_t flags = (om && ov) ? MODEL_FLAG_OPTIM : 0;
    int32_t hh[8] = { flags,
                       m->cfg.vocab_size, m->cfg.context_len, m->cfg.emb_dim,
                       m->cfg.n_layers, m->cfg.n_heads, m->cfg.pad_token_id, step };
    int ok = 1;
    ok &= fwrite(hh, sizeof(int32_t), 8, f) == 8;
    ok &= fwrite(m->params, sizeof(float), m->n_params, f) == m->n_params;
    if (flags) {
        ok &= fwrite(om, sizeof(float), m->n_params, f) == m->n_params;
        ok &= fwrite(ov, sizeof(float), m->n_params, f) == m->n_params;
    }
    fclose(f);
    return ok ? 0 : -1;
}

int model_load(Model *m, const char *path, float **om, float **ov, int *has_optim, int *step) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int32_t hh[8];
    if (fread(hh, sizeof(int32_t), 8, f) != 8) { fclose(f); return -1; }
    if (step) *step = hh[7];
    ModelConfig cfg;
    cfg.vocab_size = hh[1]; cfg.context_len = hh[2]; cfg.emb_dim = hh[3];
    cfg.n_layers = hh[4]; cfg.n_heads = hh[5]; cfg.pad_token_id = hh[6];
    Model tmp;
    if (model_init(&tmp, &cfg, 1) != 0) { fclose(f); return -1; }
    if (fread(tmp.params, sizeof(float), tmp.n_params, f) != tmp.n_params) {
        model_free(&tmp); fclose(f); return -1;
    }
    if (hh[2] & MODEL_FLAG_OPTIM) {
        if (om && ov) {
            *om = malloc(tmp.n_params * sizeof(float));
            *ov = malloc(tmp.n_params * sizeof(float));
            if (!*om || !*ov) { free(*om); free(*ov); model_free(&tmp); fclose(f); return -1; }
            if (fread(*om, sizeof(float), tmp.n_params, f) != tmp.n_params ||
                fread(*ov, sizeof(float), tmp.n_params, f) != tmp.n_params) {
                free(*om); free(*ov); *om = *ov = NULL; model_free(&tmp); fclose(f); return -1;
            }
            if (has_optim) *has_optim = 1;
        } else {
            fseek(f, (long)(2 * tmp.n_params * sizeof(float)), SEEK_CUR);
            if (has_optim) *has_optim = 0;
        }
    } else {
        if (om) *om = NULL;
        if (ov) *ov = NULL;
        if (has_optim) *has_optim = 0;
    }
    fclose(f);
    model_free(m);
    *m = tmp;
    return 0;
}

void adam_step(Model *m, float *mo, float *vo, float lr, float b1, float b2,
               float eps, int step, float clip) {
    size_t n = m->n_params;
    if (clip > 0.0f) {
        double ss = 0;
        for (size_t i = 0; i < n; i++) { double g = m->grads[i]; ss += g * g; }
        float gn = sqrtf((float)ss);
        if (gn > clip && gn > 0.0f) {
            float sc = clip / gn;
            for (size_t i = 0; i < n; i++) m->grads[i] *= sc;
        }
    }
    float bc1 = 1.0f - powf(b1, (float)step);
    float bc2 = 1.0f - powf(b2, (float)step);
    for (size_t i = 0; i < n; i++) {
        float g = m->grads[i];
        mo[i] = b1 * mo[i] + (1.0f - b1) * g;
        vo[i] = b2 * vo[i] + (1.0f - b2) * g * g;
        float mh = mo[i] / bc1;
        float vh = vo[i] / bc2;
        m->params[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}
