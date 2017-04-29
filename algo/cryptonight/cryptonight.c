/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include <stdlib.h>
#include <string.h>
#include <mm_malloc.h>

#ifndef BUILD_TEST
#   include "xmrig.h"
#endif

#include "crypto/c_groestl.h"
#include "crypto/c_blake256.h"
#include "crypto/c_jh.h"
#include "crypto/c_skein.h"
#include "cryptonight.h"
#include "options.h"


const static char test_input[76] = {
    0x03, 0x05, 0xA0, 0xDB, 0xD6, 0xBF, 0x05, 0xCF, 0x16, 0xE5, 0x03, 0xF3, 0xA6, 0x6F, 0x78, 0x00,
    0x7C, 0xBF, 0x34, 0x14, 0x43, 0x32, 0xEC, 0xBF, 0xC2, 0x2E, 0xD9, 0x5C, 0x87, 0x00, 0x38, 0x3B,
    0x30, 0x9A, 0xCE, 0x19, 0x23, 0xA0, 0x96, 0x4B, 0x00, 0x00, 0x00, 0x08, 0xBA, 0x93, 0x9A, 0x62,
    0x72, 0x4C, 0x0D, 0x75, 0x81, 0xFC, 0xE5, 0x76, 0x1E, 0x9D, 0x8A, 0x0E, 0x6A, 0x1C, 0x3F, 0x92,
    0x4F, 0xDD, 0x84, 0x93, 0xD1, 0x11, 0x56, 0x49, 0xC0, 0x5E, 0xB6, 0x01
};


const static char test_output[32] = {
    0x1A, 0x3F, 0xFB, 0xEE, 0x90, 0x9B, 0x42, 0x0D, 0x91, 0xF7, 0xBE, 0x6E, 0x5F, 0xB5, 0x6D, 0xB7,
    0x1B, 0x31, 0x10, 0xD8, 0x86, 0x01, 0x1E, 0x87, 0x7E, 0xE5, 0x78, 0x6A, 0xFD, 0x08, 0x01, 0x00
};


void cryptonight_av1_aesni(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_av4_softaes(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);

#if defined(__x86_64__)
  void cryptonight_av2_aesni_stak(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
  void cryptonight_av3_aesni_bmi2(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
  void cryptonight_av5_aesni_experimental(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
#endif

void (*cryptonight_hash_ctx)(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx) = NULL;


static bool self_test() {
    char output[32];

    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) _mm_malloc(sizeof(struct cryptonight_ctx), 16);
    ctx->memory = (uint8_t *) _mm_malloc(MEMORY, 16);

    cryptonight_hash_ctx(test_input, sizeof(test_input), output, ctx);

    _mm_free(ctx->memory);
    _mm_free(ctx);

    return memcmp(output, test_output, 32) == 0;
}


bool cryptonight_init(int variant)
{
    switch (variant) {
        case XMR_AV1_AESNI:
            cryptonight_hash_ctx = cryptonight_av1_aesni;
            break;

#       if defined(__x86_64__)
        case XMR_AV2_STAK:
            cryptonight_hash_ctx = cryptonight_av2_aesni_stak;
            break;

        case XMR_AV3_AESNI_BMI2:
            cryptonight_hash_ctx = cryptonight_av3_aesni_bmi2;
            break;

        case XMR_AV5_EXPERIMENTAL:
            cryptonight_hash_ctx = cryptonight_av5_aesni_experimental;
            break;
#       endif

        case XMR_AV4_SOFT_AES:
             cryptonight_hash_ctx = cryptonight_av4_softaes;
             break;

        default:
            break;
    }

    return self_test();
}


static inline void do_blake_hash(const void* input, size_t len, char* output) {
    blake256_hash((uint8_t*)output, input, len);
}


static inline void do_groestl_hash(const void* input, size_t len, char* output) {
    groestl(input, len * 8, (uint8_t*)output);
}


static inline void do_jh_hash(const void* input, size_t len, char* output) {
    jh_hash(32 * 8, input, 8 * len, (uint8_t*)output);
}


static inline void do_skein_hash(const void* input, size_t len, char* output) {
    skein_hash(8 * 32, input, 8 * len, (uint8_t*)output);
}


void (* const extra_hashes[4])(const void *, size_t, char *) = {do_blake_hash, do_groestl_hash, do_jh_hash, do_skein_hash};


#ifndef BUILD_TEST
int scanhash_cryptonight(int thr_id, uint32_t *hash, uint32_t *restrict blob, size_t blob_size, uint32_t target, uint32_t max_nonce, unsigned long *restrict hashes_done, struct cryptonight_ctx *restrict ctx) {
    uint32_t *nonceptr = (uint32_t*) (((char*) blob) + 39);

    do {
        cryptonight_hash_ctx(blob, blob_size, hash, ctx);
        (*hashes_done)++;

        if (unlikely(hash[7] < target)) {
            return 1;
        }

        (*nonceptr)++;
    } while (likely(((*nonceptr) < max_nonce && !work_restart[thr_id].restart)));

    return 0;
}
#endif
