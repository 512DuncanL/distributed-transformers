#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

/* ------------------------------------------------------------------ */
/*  Model: a small decoder-only transformer LM.                       */
/*                                                                     */
/*  - All parameters live in one contiguous float buffer (m->params).  */
/*  - Gradients live in a parallel buffer (m->grads).                  */
/*  - Matrix multiplies use OpenBLAS cblas_sgemm (row-major).          */
/* ------------------------------------------------------------------ */

#define MODEL_MAGIC    0x4D4F444C   /* "MODL" */
#define MODEL_VERSION  1
#define MODEL_FLAG_OPTIM 0x1        /* checkpoint also contains Adam m/v */

typedef struct {
    int vocab_size;
    int context_len;   /* max sequence length L (positions 0..context_len-1) */
    int emb_dim;       /* D */
    int n_layers;      /* number of transformer blocks */
    int n_heads;       /* H; must divide emb_dim */
    int pad_token_id;  /* token id used for padding; masked out of the loss */
} ModelConfig;

typedef struct {
    ModelConfig cfg;
    int head_dim;      /* emb_dim / n_heads */
    int mlp_hidden;    /* 4 * emb_dim (SwiGLU intermediate) */
    size_t n_params;   /* number of floats in params[] (and grads[]) */
    float *params;     /* flat contiguous parameter buffer */
    float *grads;      /* flat contiguous gradient buffer */
    void *priv;        /* opaque workspace + layer pointers (model.c) */
} Model;

/* ------------------------------------------------------------------ */
/*  Geometry                                                          */
/* ------------------------------------------------------------------ */
/* Fills head_dim, mlp_hidden, n_params for a config. Returns 0 on
 * success, -1 if the config is invalid (e.g. emb_dim % n_heads != 0). */
int model_compute_geometry(const ModelConfig *cfg,
                           int *head_dim, int *mlp_hidden, size_t *n_params);

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */
/* Allocates and initializes params/grads with a deterministic RNG
 * seeded by `seed`. Returns 0 on success, -1 on error. */
int  model_init(Model *m, const ModelConfig *cfg, unsigned int seed);
void model_free(Model *m);

/* ------------------------------------------------------------------ */
/*  Forward                                                           */
/* ------------------------------------------------------------------ */
/* Computes logits [B * L * vocab_size] from token ids [B * L].       */
void model_forward(const Model *m, const int *ids, int B, int L, float *logits);

/* ------------------------------------------------------------------ */
/*  Forward + loss (+ optional backward)                              */
/* ------------------------------------------------------------------ */
/* Runs forward, computes mean cross-entropy over non-PAD targets
 * (target = ids shifted by one position), and if compute_grad != 0
 * also backpropagates, filling m->grads. `logits` may be NULL.
 * Returns the loss. */
float model_train_step(Model *m, const int *ids, int B, int L,
                       float *logits, int compute_grad);

/* ------------------------------------------------------------------ */
/*  Persistence                                                       */
/* ------------------------------------------------------------------ */
/* Save weights (and, if optim_m/optim_v are non-NULL, Adam state) plus the
 * global training step count to `path`. Returns 0 on success, -1 on error. */
int model_save(const Model *m, const char *path,
               const float *optim_m, const float *optim_v, int step);

/* Load a model from `path`. If the checkpoint contains Adam state,
 * allocates *optim_m / *optim_v (caller frees) and sets *has_optim = 1.
 * Pass NULL for optim_m/optim_v to skip them. *step (if non-NULL) receives
 * the saved step count. Returns 0 on success. */
int model_load(Model *m, const char *path,
               float **optim_m, float **optim_v, int *has_optim, int *step);

/* ------------------------------------------------------------------ */
/*  Optimizer                                                         */
/* ------------------------------------------------------------------ */
/* One Adam step over m->params using m->grads. m_opt/v_opt are
 * per-parameter moment buffers of length m->n_params. step is the
 * global step count (>= 1). Gradients are clipped by global norm
 * `grad_clip` first (0 disables clipping). */
void adam_step(Model *m, float *m_opt, float *v_opt,
               float lr, float beta1, float beta2, float eps,
               int step, float grad_clip);

#endif /* MODEL_H */
