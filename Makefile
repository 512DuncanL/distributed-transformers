CC = gcc
CFLAGS = -Wall -Wextra -O2
MODEL_CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math
BLAS_LIBS = -lopenblas -lpthread -lm

all: build_tokenizer tokenize train generate

# Test binary: removed (not needed for training/inference)

# Training binary: build_tokenizer.c only
build_tokenizer: build_tokenizer.o
	$(CC) $(CFLAGS) -o build_tokenizer build_tokenizer.o

# Encode/decode binary: needs tokenizer.h, tokenize.c only
tokenize: tokenize.o
	$(CC) $(CFLAGS) -o tokenize tokenize.o

# Transformer training / inference: model.c + tokenizer library + OpenBLAS
train: train.o model.o tokenize_lib.o
	$(CC) $(MODEL_CFLAGS) -o train train.o model.o tokenize_lib.o $(BLAS_LIBS)

generate: generate.o model.o tokenize_lib.o
	$(CC) $(MODEL_CFLAGS) -o generate generate.o model.o tokenize_lib.o $(BLAS_LIBS)

test_model: test_model.c model.c model.h
	$(CC) $(MODEL_CFLAGS) -o test_model test_model.c model.c $(BLAS_LIBS)

build_tokenizer.o: build_tokenizer.c tokenizer.h
	$(CC) $(CFLAGS) -c build_tokenizer.c -o build_tokenizer.o

tokenize.o: tokenize.c tokenizer.h
	$(CC) $(CFLAGS) -c tokenize.c -o tokenize.o

# tokenizer library object for train/generate (its CLI main is renamed away)
tokenize_lib.o: tokenize.c tokenizer.h
	$(CC) $(CFLAGS) -Dmain=tokenize_main -c tokenize.c -o tokenize_lib.o

model.o: model.c model.h
	$(CC) $(MODEL_CFLAGS) -c model.c -o model.o

train.o: train.c model.h tokenizer.h
	$(CC) $(MODEL_CFLAGS) -c train.c -o train.o

generate.o: generate.c model.h tokenizer.h
	$(CC) $(MODEL_CFLAGS) -c generate.c -o generate.o

clean:
	rm -f build_tokenizer tokenize train generate test_model *.o

# NOTE: tokenizer.bin, ckpt.bin, weights.bin are intentionally NOT removed by
# `clean` — they are expensive data/training artifacts, not build products.

.PHONY: all clean
