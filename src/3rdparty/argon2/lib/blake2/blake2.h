#ifndef ARGON2_BLAKE2_H
#define ARGON2_BLAKE2_H

#include <stddef.h>
#include <stdint.h>

enum blake2b_constant {
    BLAKE2B_BLOCKBYTES = 128,
    BLAKE2B_OUTBYTES = 64,
    BLAKE2B_KEYBYTES = 64,
    BLAKE2B_SALTBYTES = 16,
    BLAKE2B_PERSONALBYTES = 16
};

typedef struct __blake2b_state {
    uint64_t h[8];
    uint64_t t[2];
    uint8_t buf[BLAKE2B_BLOCKBYTES];
    size_t buflen;
} blake2b_state;

/* Streaming API */
void xmrig_ar2_blake2b_init(blake2b_state *S, size_t outlen);
void xmrig_ar2_blake2b_update(blake2b_state *S, const void *in, size_t inlen);
void xmrig_ar2_blake2b_final(blake2b_state *S, void *out, size_t outlen);

void xmrig_ar2_blake2b_long(void *out, size_t outlen, const void *in, size_t inlen);

#endif // ARGON2_BLAKE2_H

