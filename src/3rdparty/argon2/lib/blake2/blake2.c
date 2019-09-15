#include <string.h>

#include "blake2/blake2.h"
#include "blake2/blake2-impl.h"

#include "core.h"

static const uint64_t blake2b_IV[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)
};

#define rotr64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

static const unsigned int blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

#define G(m, r, i, a, b, c, d) \
    do { \
        a = a + b + m[blake2b_sigma[r][2 * i + 0]]; \
        d = rotr64(d ^ a, 32); \
        c = c + d; \
        b = rotr64(b ^ c, 24); \
        a = a + b + m[blake2b_sigma[r][2 * i + 1]]; \
        d = rotr64(d ^ a, 16); \
        c = c + d; \
        b = rotr64(b ^ c, 63); \
    } while ((void)0, 0)

#define ROUND(m, v, r) \
    do { \
        G(m, r, 0, v[0], v[4], v[ 8], v[12]); \
        G(m, r, 1, v[1], v[5], v[ 9], v[13]); \
        G(m, r, 2, v[2], v[6], v[10], v[14]); \
        G(m, r, 3, v[3], v[7], v[11], v[15]); \
        G(m, r, 4, v[0], v[5], v[10], v[15]); \
        G(m, r, 5, v[1], v[6], v[11], v[12]); \
        G(m, r, 6, v[2], v[7], v[ 8], v[13]); \
        G(m, r, 7, v[3], v[4], v[ 9], v[14]); \
    } while ((void)0, 0)

void blake2b_compress(blake2b_state *S, const void *block, uint64_t f0)
{
    uint64_t m[16];
    uint64_t v[16];

    m[ 0] = load64((const uint64_t *)block +  0);
    m[ 1] = load64((const uint64_t *)block +  1);
    m[ 2] = load64((const uint64_t *)block +  2);
    m[ 3] = load64((const uint64_t *)block +  3);
    m[ 4] = load64((const uint64_t *)block +  4);
    m[ 5] = load64((const uint64_t *)block +  5);
    m[ 6] = load64((const uint64_t *)block +  6);
    m[ 7] = load64((const uint64_t *)block +  7);
    m[ 8] = load64((const uint64_t *)block +  8);
    m[ 9] = load64((const uint64_t *)block +  9);
    m[10] = load64((const uint64_t *)block + 10);
    m[11] = load64((const uint64_t *)block + 11);
    m[12] = load64((const uint64_t *)block + 12);
    m[13] = load64((const uint64_t *)block + 13);
    m[14] = load64((const uint64_t *)block + 14);
    m[15] = load64((const uint64_t *)block + 15);

    v[ 0] = S->h[0];
    v[ 1] = S->h[1];
    v[ 2] = S->h[2];
    v[ 3] = S->h[3];
    v[ 4] = S->h[4];
    v[ 5] = S->h[5];
    v[ 6] = S->h[6];
    v[ 7] = S->h[7];
    v[ 8] = blake2b_IV[0];
    v[ 9] = blake2b_IV[1];
    v[10] = blake2b_IV[2];
    v[11] = blake2b_IV[3];
    v[12] = blake2b_IV[4] ^ S->t[0];
    v[13] = blake2b_IV[5] ^ S->t[1];
    v[14] = blake2b_IV[6] ^ f0;
    v[15] = blake2b_IV[7];

    ROUND(m, v, 0);
    ROUND(m, v, 1);
    ROUND(m, v, 2);
    ROUND(m, v, 3);
    ROUND(m, v, 4);
    ROUND(m, v, 5);
    ROUND(m, v, 6);
    ROUND(m, v, 7);
    ROUND(m, v, 8);
    ROUND(m, v, 9);
    ROUND(m, v, 10);
    ROUND(m, v, 11);

    S->h[0] ^= v[0] ^ v[ 8];
    S->h[1] ^= v[1] ^ v[ 9];
    S->h[2] ^= v[2] ^ v[10];
    S->h[3] ^= v[3] ^ v[11];
    S->h[4] ^= v[4] ^ v[12];
    S->h[5] ^= v[5] ^ v[13];
    S->h[6] ^= v[6] ^ v[14];
    S->h[7] ^= v[7] ^ v[15];
}

static void blake2b_increment_counter(blake2b_state *S, uint64_t inc)
{
    S->t[0] += inc;
    S->t[1] += (S->t[0] < inc);
}

static void blake2b_init_state(blake2b_state *S)
{
    memcpy(S->h, blake2b_IV, sizeof(S->h));
    S->t[1] = S->t[0] = 0;
    S->buflen = 0;
}

void blake2b_init(blake2b_state *S, size_t outlen)
{
    blake2b_init_state(S);
    /* XOR initial state with param block: */
    S->h[0] ^= (uint64_t)outlen | (UINT64_C(1) << 16) | (UINT64_C(1) << 24);
}

void blake2b_update(blake2b_state *S, const void *in, size_t inlen)
{
    const uint8_t *pin = (const uint8_t *)in;

    if (S->buflen + inlen > BLAKE2B_BLOCKBYTES) {
        size_t left = S->buflen;
        size_t fill = BLAKE2B_BLOCKBYTES - left;
        memcpy(&S->buf[left], pin, fill);
        blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
        blake2b_compress(S, S->buf, 0);
        S->buflen = 0;
        inlen -= fill;
        pin += fill;
        /* Avoid buffer copies when possible */
        while (inlen > BLAKE2B_BLOCKBYTES) {
            blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
            blake2b_compress(S, pin, 0);
            inlen -= BLAKE2B_BLOCKBYTES;
            pin += BLAKE2B_BLOCKBYTES;
        }
    }
    memcpy(&S->buf[S->buflen], pin, inlen);
    S->buflen += inlen;
}

void blake2b_final(blake2b_state *S, void *out, size_t outlen)
{
    uint8_t buffer[BLAKE2B_OUTBYTES] = {0};
    unsigned int i;

    blake2b_increment_counter(S, S->buflen);
    memset(&S->buf[S->buflen], 0, BLAKE2B_BLOCKBYTES - S->buflen); /* Padding */
    blake2b_compress(S, S->buf, UINT64_C(0xFFFFFFFFFFFFFFFF));

    for (i = 0; i < 8; ++i) { /* Output full hash to temp buffer */
        store64(buffer + i * sizeof(uint64_t), S->h[i]);
    }

    memcpy(out, buffer, outlen);
    clear_internal_memory(buffer, sizeof(buffer));
    clear_internal_memory(S->buf, sizeof(S->buf));
    clear_internal_memory(S->h, sizeof(S->h));
}

void blake2b_long(void *out, size_t outlen, const void *in, size_t inlen)
{
    uint8_t *pout = (uint8_t *)out;
    blake2b_state blake_state;
    uint8_t outlen_bytes[sizeof(uint32_t)] = {0};

    store32(outlen_bytes, (uint32_t)outlen);
    if (outlen <= BLAKE2B_OUTBYTES) {
        blake2b_init(&blake_state, outlen);
        blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes));
        blake2b_update(&blake_state, in, inlen);
        blake2b_final(&blake_state, pout, outlen);
    } else {
        uint32_t toproduce;
        uint8_t out_buffer[BLAKE2B_OUTBYTES];

        blake2b_init(&blake_state, BLAKE2B_OUTBYTES);
        blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes));
        blake2b_update(&blake_state, in, inlen);
        blake2b_final(&blake_state, out_buffer, BLAKE2B_OUTBYTES);

        memcpy(pout, out_buffer, BLAKE2B_OUTBYTES / 2);
        pout += BLAKE2B_OUTBYTES / 2;
        toproduce = (uint32_t)outlen - BLAKE2B_OUTBYTES / 2;

        while (toproduce > BLAKE2B_OUTBYTES) {
            blake2b_init(&blake_state, BLAKE2B_OUTBYTES);
            blake2b_update(&blake_state, out_buffer, BLAKE2B_OUTBYTES);
            blake2b_final(&blake_state, out_buffer, BLAKE2B_OUTBYTES);

            memcpy(pout, out_buffer, BLAKE2B_OUTBYTES / 2);
            pout += BLAKE2B_OUTBYTES / 2;
            toproduce -= BLAKE2B_OUTBYTES / 2;
        }

        blake2b_init(&blake_state, toproduce);
        blake2b_update(&blake_state, out_buffer, BLAKE2B_OUTBYTES);
        blake2b_final(&blake_state, out_buffer, toproduce);

        memcpy(pout, out_buffer, toproduce);

        clear_internal_memory(out_buffer, sizeof(out_buffer));
    }
}
