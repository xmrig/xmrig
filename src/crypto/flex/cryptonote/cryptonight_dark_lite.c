// Copyright (c) 2012-2013 The Cryptonote developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.
// Portions Copyright (c) 2018 The Monero developers
// Portions Copyright (c) 2018 The darkCoin Developers

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "crypto/oaes_lib.h"
#include "crypto/c_keccak.h"
#include "crypto/c_groestl.h"
#include "crypto/c_blake256.h"
#include "crypto/c_jh.h"
#include "../../cn/c_skein.h"
#include "crypto/int-util.h"
#include "crypto/hash-ops.h"
#include "crypto/variant2_int_sqrt.h"

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#define MEMORY          524288 /* 512KB - 2^19 */
#define ITER            262144 /* 2^18 */
#define ITER_DIV        131072 /* 2^17 */
#define AES_BLOCK_SIZE  16
#define AES_KEY_SIZE    32 /*16*/
#define INIT_SIZE_BLK   8
#define INIT_SIZE_BYTE  (INIT_SIZE_BLK * AES_BLOCK_SIZE)
#define CN_INIT         (MEMORY / INIT_SIZE_BYTE)
#define CN_AES_INIT     (MEMORY / AES_BLOCK_SIZE) / 2

#define VARIANT1_1(p) \
  do if (variant == 1) \
  { \
    const uint8_t tmp = ((const uint8_t*)(p))[11]; \
    static const uint32_t table = 0x75310; \
    const uint8_t index = (((tmp >> 3) & 6) | (tmp & 1)) << 1; \
    ((uint8_t*)(p))[11] = tmp ^ ((table >> index) & 0x30); \
  } while(0)

#define VARIANT1_2(p) \
   do if (variant == 1) \
   { \
     ((uint64_t*)p)[1] ^= tweak1_2; \
   } while(0)

#define VARIANT1_INIT() \
  if (variant == 1 && len < 43) \
  { \
    fprintf(stderr, "Cryptonight variant 1 needs at least 43 bytes of data"); \
    _exit(1); \
  } \
  const uint64_t tweak1_2 = (variant == 1) ? *(const uint64_t*)(((const uint8_t*)input)+35) ^ ctx->state.hs.w[24] : 0

#define U64(p) ((uint64_t*)(p))

#define VARIANT2_INIT(b, state) \
  uint64_t division_result; \
  uint64_t sqrt_result; \
  do if (variant >= 2) \
  { \
    U64(b)[2] = state.hs.w[8] ^ state.hs.w[10]; \
    U64(b)[3] = state.hs.w[9] ^ state.hs.w[11]; \
    division_result = state.hs.w[12]; \
    sqrt_result = state.hs.w[13]; \
  } while (0)

#define VARIANT2_SHUFFLE_ADD(base_ptr, offset, a, b) \
  do if (variant >= 2) \
  { \
    uint64_t* chunk1 = U64((base_ptr) + ((offset) ^ 0x10)); \
    uint64_t* chunk2 = U64((base_ptr) + ((offset) ^ 0x20)); \
    uint64_t* chunk3 = U64((base_ptr) + ((offset) ^ 0x30)); \
    \
    const uint64_t chunk1_old[2] = { chunk1[0], chunk1[1] }; \
    \
    chunk1[0] = chunk3[0] + U64(b + 16)[0]; \
    chunk1[1] = chunk3[1] + U64(b + 16)[1]; \
    \
    chunk3[0] = chunk2[0] + U64(a)[0]; \
    chunk3[1] = chunk2[1] + U64(a)[1]; \
    \
    chunk2[0] = chunk1_old[0] + U64(b)[0]; \
    chunk2[1] = chunk1_old[1] + U64(b)[1]; \
  } while (0)

#define VARIANT2_INTEGER_MATH_DIVISION_STEP(b, ptr) \
  ((uint64_t*)(b))[0] ^= division_result ^ (sqrt_result << 32); \
  { \
    const uint64_t dividend = ((uint64_t*)(ptr))[1]; \
    const uint32_t divisor = (((uint32_t*)(ptr))[0] + (uint32_t)(sqrt_result << 1)) | 0x80000001UL; \
    division_result = ((uint32_t)(dividend / divisor)) + \
                     (((uint64_t)(dividend % divisor)) << 32); \
  } \
  const uint64_t sqrt_input = ((uint64_t*)(ptr))[0] + division_result

#define VARIANT2_INTEGER_MATH(b, ptr) \
    do if (variant >= 2) \
    { \
      VARIANT2_INTEGER_MATH_DIVISION_STEP(b, ptr); \
      VARIANT2_INTEGER_MATH_SQRT_STEP_FP64(); \
      VARIANT2_INTEGER_MATH_SQRT_FIXUP(sqrt_result); \
    } while (0)

#define VARIANT2_2() \
  do if (variant >= 2) { \
    ((uint64_t*)(ctx->long_state + ((j * AES_BLOCK_SIZE) ^ 0x10)))[0] ^= hi; \
    ((uint64_t*)(ctx->long_state + ((j * AES_BLOCK_SIZE) ^ 0x10)))[1] ^= lo; \
    hi ^= ((uint64_t*)(ctx->long_state + ((j * AES_BLOCK_SIZE) ^ 0x20)))[0]; \
    lo ^= ((uint64_t*)(ctx->long_state + ((j * AES_BLOCK_SIZE) ^ 0x20)))[1]; \
  } while (0)

#pragma pack(push, 1)
union cn_slow_hash_state {
    union hash_state hs;
    struct {
        uint8_t k[64];
        uint8_t init[INIT_SIZE_BYTE];
    };
};
#pragma pack(pop)

static void do_dark_lite_blake_hash(const void* input, size_t len, char* output) {
    blake256_hash((uint8_t*)output, input, len);
}

void do_dark_lite_groestl_hash(const void* input, size_t len, char* output) {
    groestl(input, len * 8, (uint8_t*)output);
}

static void do_dark_lite_jh_hash(const void* input, size_t len, char* output) {
    int r = jh_hash(HASH_SIZE * 8, input, 8 * len, (uint8_t*)output);
    assert(SUCCESS == r);
}

static void do_dark_lite_skein_hash(const void* input, size_t len, char* output) {
    int r = skein_hash(8 * HASH_SIZE, input, 8 * len, (uint8_t*)output);
    assert(SKEIN_SUCCESS == r);
}

static void (* const extra_hashes[3])(const void *, size_t, char *) = {
    do_dark_lite_blake_hash, do_dark_lite_groestl_hash, do_dark_lite_skein_hash
};

extern int aesb_single_round(const uint8_t *in, uint8_t*out, const uint8_t *expandedKey);
extern int aesb_pseudo_round(const uint8_t *in, uint8_t *out, const uint8_t *expandedKey);

static inline size_t e2i(const uint8_t* a) {
    return (*((uint64_t*) a) / AES_BLOCK_SIZE) & (CN_AES_INIT - 1);
}

static void mul(const uint8_t* a, const uint8_t* b, uint8_t* res) {
    ((uint64_t*) res)[1] = mul128(((uint64_t*) a)[0], ((uint64_t*) b)[0], (uint64_t*) res);
}

static void sum_half_blocks(uint8_t* a, const uint8_t* b) {
    uint64_t a0, a1, b0, b1;

    a0 = SWAP64LE(((uint64_t*) a)[0]);
    a1 = SWAP64LE(((uint64_t*) a)[1]);
    b0 = SWAP64LE(((uint64_t*) b)[0]);
    b1 = SWAP64LE(((uint64_t*) b)[1]);
    a0 += b0;
    a1 += b1;
    ((uint64_t*) a)[0] = SWAP64LE(a0);
    ((uint64_t*) a)[1] = SWAP64LE(a1);
}

static inline void copy_block(uint8_t* dst, const uint8_t* src) {
    ((uint64_t*) dst)[0] = ((uint64_t*) src)[0];
    ((uint64_t*) dst)[1] = ((uint64_t*) src)[1];
}

static void swap_blocks(uint8_t* a, uint8_t* b) {
    size_t i;
    uint8_t t;
    for (i = 0; i < AES_BLOCK_SIZE; i++) {
        t = a[i];
        a[i] = b[i];
        b[i] = t;
    }
}

static inline void xor_blocks(uint8_t* a, const uint8_t* b) {
    ((uint64_t*) a)[0] ^= ((uint64_t*) b)[0];
    ((uint64_t*) a)[1] ^= ((uint64_t*) b)[1];
}

static inline void xor_blocks_dst(const uint8_t* a, const uint8_t* b, uint8_t* dst) {
    ((uint64_t*) dst)[0] = ((uint64_t*) a)[0] ^ ((uint64_t*) b)[0];
    ((uint64_t*) dst)[1] = ((uint64_t*) a)[1] ^ ((uint64_t*) b)[1];
}

struct cryptonightdarklite_ctx {
    uint8_t long_state[MEMORY];
    union cn_slow_hash_state state;
    uint8_t text[INIT_SIZE_BYTE];
    uint8_t a[AES_BLOCK_SIZE];
    uint8_t b[AES_BLOCK_SIZE * 2];
    uint8_t c[AES_BLOCK_SIZE];
    uint8_t aes_key[AES_KEY_SIZE];
    oaes_ctx* aes_ctx;
};

void cryptonightdarklite_hash(const char* input, char* output, uint32_t len, int variant) {
#if defined(_MSC_VER)
    struct cryptonightdarklite_ctx *ctx = _malloca(sizeof(struct cryptonightdarklite_ctx));
#else
    struct cryptonightdarklite_ctx *ctx = alloca(sizeof(struct cryptonightdarklite_ctx));
#endif
    hash_process(&ctx->state.hs, (const uint8_t*) input, len);
    memcpy(ctx->text, ctx->state.init, INIT_SIZE_BYTE);
    memcpy(ctx->aes_key, ctx->state.hs.b, AES_KEY_SIZE);
    ctx->aes_ctx = (oaes_ctx*) oaes_alloc();
    size_t i, j;

    VARIANT1_INIT();
    VARIANT2_INIT(ctx->b, ctx->state);

    oaes_key_import_data(ctx->aes_ctx, ctx->aes_key, AES_KEY_SIZE);
    for (i = 0; i < CN_INIT; i++) {
        for (j = 0; j < INIT_SIZE_BLK; j++) {
            aesb_pseudo_round(&ctx->text[AES_BLOCK_SIZE * j],
                    &ctx->text[AES_BLOCK_SIZE * j],
                    ctx->aes_ctx->key->exp_data);
        }
        memcpy(&ctx->long_state[i * INIT_SIZE_BYTE], ctx->text, INIT_SIZE_BYTE);
    }

    for (i = 0; i < 16; i++) {
        ctx->a[i] = ctx->state.k[i] ^ ctx->state.k[32 + i];
        ctx->b[i] = ctx->state.k[16 + i] ^ ctx->state.k[48 + i];
    }

    for (i = 0; i < ITER_DIV; i++) {
        /* Dependency chain: address -> read value ------+
         * written value <-+ hard function (AES or MUL) <+
         * next address  <-+
         */
        /* Iteration 1 */
        j = e2i(ctx->a);
        aesb_single_round(&ctx->long_state[j * AES_BLOCK_SIZE], ctx->c, ctx->a);
        VARIANT2_SHUFFLE_ADD(ctx->long_state, j * AES_BLOCK_SIZE, ctx->a, ctx->b);
        xor_blocks_dst(ctx->c, ctx->b, &ctx->long_state[j * AES_BLOCK_SIZE]);
        VARIANT1_1((uint8_t*)&ctx->long_state[j * AES_BLOCK_SIZE]);
        /* Iteration 2 */
        j = e2i(ctx->c);

        uint64_t* dst = (uint64_t*)&ctx->long_state[j * AES_BLOCK_SIZE];

        uint64_t t[2];
        t[0] = dst[0];
        t[1] = dst[1];

        VARIANT2_INTEGER_MATH(t, ctx->c);

        uint64_t hi;
        uint64_t lo = mul128(((uint64_t*)ctx->c)[0], t[0], &hi);

        VARIANT2_2();
        VARIANT2_SHUFFLE_ADD(ctx->long_state, j * AES_BLOCK_SIZE, ctx->a, ctx->b);

        ((uint64_t*)ctx->a)[0] += hi;
        ((uint64_t*)ctx->a)[1] += lo;

        dst[0] = ((uint64_t*)ctx->a)[0];
        dst[1] = ((uint64_t*)ctx->a)[1];

        ((uint64_t*)ctx->a)[0] ^= t[0];
        ((uint64_t*)ctx->a)[1] ^= t[1];

        VARIANT1_2((uint8_t*)&ctx->long_state[j * AES_BLOCK_SIZE]);
        copy_block(ctx->b + AES_BLOCK_SIZE, ctx->b);
        copy_block(ctx->b, ctx->c);
    }

    memcpy(ctx->text, ctx->state.init, INIT_SIZE_BYTE);
    oaes_key_import_data(ctx->aes_ctx, &ctx->state.hs.b[32], AES_KEY_SIZE);
    for (i = 0; i < CN_INIT; i++) {
        for (j = 0; j < INIT_SIZE_BLK; j++) {
            xor_blocks(&ctx->text[j * AES_BLOCK_SIZE],
                    &ctx->long_state[i * INIT_SIZE_BYTE + j * AES_BLOCK_SIZE]);
            aesb_pseudo_round(&ctx->text[j * AES_BLOCK_SIZE],
                    &ctx->text[j * AES_BLOCK_SIZE],
                    ctx->aes_ctx->key->exp_data);
        }
    }
    memcpy(ctx->state.init, ctx->text, INIT_SIZE_BYTE);
    hash_permutation(&ctx->state.hs);
    /*memcpy(hash, &state, 32);*/
    extra_hashes[ctx->state.hs.b[0] & 2](&ctx->state, 200, output);
    oaes_free((OAES_CTX **) &ctx->aes_ctx);
}

void cryptonightdarklite_fast_hash(const char* input, char* output, uint32_t len) {
    union hash_state state;
    hash_process(&state, (const uint8_t*) input, len);
    memcpy(output, &state, HASH_SIZE);
}
