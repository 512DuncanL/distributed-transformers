#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */
#define MAX_VOCAB 4096
#define MAX_TOKEN_LEN 64
#define MAX_TRIE_NODES 65536

/* Special tokens occupy IDs 0..NUM_SPECIAL-1; the 256 single-byte base tokens
 * follow at IDs NUM_SPECIAL..NUM_SPECIAL+255, so any byte maps to its base
 * token by simple arithmetic (NUM_SPECIAL + byte). */
#define NUM_SPECIAL 3
#define TOK_EOS 0
#define TOK_PAD 1
#define TOK_BOS 2
#define BASE_VOCAB (NUM_SPECIAL + 256)

/* ------------------------------------------------------------------ */
/*  Trie node for BPE encoding (greedy longest-match)                 */
/* ------------------------------------------------------------------ */
typedef struct TrieNode {
    int token_id;              /* -1 if not a complete token, >=0 if it is */
    int children[256];         /* child indices, -1 = no child */
} TrieNode;

/* The trie itself is only used at encode time; its global storage lives in
 * tokenize.c rather than here, so translation units that don't tokenize (e.g.
 * build_tokenizer.c) don't carry an unused copy. */

/* ------------------------------------------------------------------ */
/*  Tokenizer struct                                                  */
/* ------------------------------------------------------------------ */
typedef struct {
    char tokens[MAX_VOCAB][MAX_TOKEN_LEN];  /* token string at each ID */
    int  vocab_size;
    int  max_token_len;

    /* Trie pointers (global storage) */
    TrieNode *trie;
    int trie_root;
} Tokenizer;

/* ------------------------------------------------------------------ */
/*  Public API — training                                             */
/* ------------------------------------------------------------------ */
void init_tokenizer(Tokenizer *t);
void build_base_vocab(Tokenizer *t);
void build_tokenizer(Tokenizer *t, char **rows, int row_count);
void write_tokenizer(const Tokenizer *t, const char *path);
void free_tokenizer(Tokenizer *t);
int  load_tokenizer(const char *path, Tokenizer *t);

/* ------------------------------------------------------------------ */
/*  Public API — encoding / decoding                                  */
/* ------------------------------------------------------------------ */
int *encode(const Tokenizer *t, const char *text, int *out_len);
char *decode(const Tokenizer *t, const int *ids, int len, int *out_len);

/* ------------------------------------------------------------------ */
/*  Utilities                                                         */
/* ------------------------------------------------------------------ */
char **split_str(char *original, int *arr_length);
void free_rows(char **rows, int count);

#endif /* TOKENIZER_H */
