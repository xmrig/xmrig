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


const static char test_input[152] = {
    0x01, 0x00, 0xFB, 0x8E, 0x8A, 0xC8, 0x05, 0x89, 0x93, 0x23, 0x37, 0x1B, 0xB7, 0x90, 0xDB, 0x19,
    0x21, 0x8A, 0xFD, 0x8D, 0xB8, 0xE3, 0x75, 0x5D, 0x8B, 0x90, 0xF3, 0x9B, 0x3D, 0x55, 0x06, 0xA9,
    0xAB, 0xCE, 0x4F, 0xA9, 0x12, 0x24, 0x45, 0x00, 0x00, 0x00, 0x00, 0xEE, 0x81, 0x46, 0xD4, 0x9F,
    0xA9, 0x3E, 0xE7, 0x24, 0xDE, 0xB5, 0x7D, 0x12, 0xCB, 0xC6, 0xC6, 0xF3, 0xB9, 0x24, 0xD9, 0x46,
    0x12, 0x7C, 0x7A, 0x97, 0x41, 0x8F, 0x93, 0x48, 0x82, 0x8F, 0x0F, 0x02,
    0x03, 0x05, 0xA0, 0xDB, 0xD6, 0xBF, 0x05, 0xCF, 0x16, 0xE5, 0x03, 0xF3, 0xA6, 0x6F, 0x78, 0x00,
    0x7C, 0xBF, 0x34, 0x14, 0x43, 0x32, 0xEC, 0xBF, 0xC2, 0x2E, 0xD9, 0x5C, 0x87, 0x00, 0x38, 0x3B,
    0x30, 0x9A, 0xCE, 0x19, 0x23, 0xA0, 0x96, 0x4B, 0x00, 0x00, 0x00, 0x08, 0xBA, 0x93, 0x9A, 0x62,
    0x72, 0x4C, 0x0D, 0x75, 0x81, 0xFC, 0xE5, 0x76, 0x1E, 0x9D, 0x8A, 0x0E, 0x6A, 0x1C, 0x3F, 0x92,
    0x4F, 0xDD, 0x84, 0x93, 0xD1, 0x11, 0x56, 0x49, 0xC0, 0x5E, 0xB6, 0x01
};


const static char test_output0[64] = {
    0x1B, 0x60, 0x6A, 0x3F, 0x4A, 0x07, 0xD6, 0x48, 0x9A, 0x1B, 0xCD, 0x07, 0x69, 0x7B, 0xD1, 0x66,
    0x96, 0xB6, 0x1C, 0x8A, 0xE9, 0x82, 0xF6, 0x1A, 0x90, 0x16, 0x0F, 0x4E, 0x52, 0x82, 0x8A, 0x7F,
    0x1A, 0x3F, 0xFB, 0xEE, 0x90, 0x9B, 0x42, 0x0D, 0x91, 0xF7, 0xBE, 0x6E, 0x5F, 0xB5, 0x6D, 0xB7,
    0x1B, 0x31, 0x10, 0xD8, 0x86, 0x01, 0x1E, 0x87, 0x7E, 0xE5, 0x78, 0x6A, 0xFD, 0x08, 0x01, 0x00
};


void cryptonight_av1_aesni(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_av2_aesni_double(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_av3_softaes(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_av4_softaes_double(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);

#ifndef XMRIG_NO_AEON
const static char test_output1[64] = {
    0x28, 0xA2, 0x2B, 0xAD, 0x3F, 0x93, 0xD1, 0x40, 0x8F, 0xCA, 0x47, 0x2E, 0xB5, 0xAD, 0x1C, 0xBE,
    0x75, 0xF2, 0x1D, 0x05, 0x3C, 0x8C, 0xE5, 0xB3, 0xAF, 0x10, 0x5A, 0x57, 0x71, 0x3E, 0x21, 0xDD,
    0x36, 0x95, 0xB4, 0xB5, 0x3B, 0xB0, 0x03, 0x58, 0xB0, 0xAD, 0x38, 0xDC, 0x16, 0x0F, 0xEB, 0x9E,
    0x00, 0x4E, 0xEC, 0xE0, 0x9B, 0x83, 0xA7, 0x2E, 0xF6, 0xBA, 0x98, 0x64, 0xD3, 0x51, 0x0C, 0x88,
};

void cryptonight_lite_av1_aesni(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_lite_av2_aesni_double(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_lite_av3_softaes(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_lite_av4_softaes_double(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
#endif

void (*cryptonight_hash_ctx)(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx) = NULL;


static bool self_test() {
    if (cryptonight_hash_ctx == NULL) {
        return false;
    }

    char output[64];

    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) _mm_malloc(sizeof(struct cryptonight_ctx), 16);
    ctx->memory = (uint8_t *) _mm_malloc(MEMORY * 2, 16);

    cryptonight_hash_ctx(test_input, 76, output, ctx);

    _mm_free(ctx->memory);
    _mm_free(ctx);

#   ifndef XMRIG_NO_AEON
    if (opt_algo == ALGO_CRYPTONIGHT_LITE) {
        return memcmp(output, test_output1, (opt_double_hash ? 64 : 32)) == 0;
    }
#   endif

    return memcmp(output, test_output0, (opt_double_hash ? 64 : 32)) == 0;
}


#ifndef XMRIG_NO_AEON
bool cryptonight_lite_init(int variant) {
    switch (variant) {
        case AEON_AV1_AESNI:
            cryptonight_hash_ctx = cryptonight_lite_av1_aesni;
            break;

        case AEON_AV2_AESNI_DOUBLE:
            opt_double_hash = true;
            cryptonight_hash_ctx = cryptonight_lite_av2_aesni_double;
            break;

        case AEON_AV3_SOFT_AES:
            cryptonight_hash_ctx = cryptonight_lite_av3_softaes;
            break;

        case AEON_AV4_SOFT_AES_DOUBLE:
            opt_double_hash = true;
            cryptonight_hash_ctx = cryptonight_lite_av4_softaes_double;
            break;

        default:
            break;
    }

    return self_test();
}
#endif


bool cryptonight_init(int variant)
{
#   ifndef XMRIG_NO_AEON
    if (opt_algo == ALGO_CRYPTONIGHT_LITE) {
        return cryptonight_lite_init(variant);
    }
#   endif

    switch (variant) {
        case XMR_AV1_AESNI:
            cryptonight_hash_ctx = cryptonight_av1_aesni;
            break;

        case XMR_AV2_AESNI_DOUBLE:
            opt_double_hash = true;
            cryptonight_hash_ctx = cryptonight_av2_aesni_double;
            break;

        case XMR_AV3_SOFT_AES:
            cryptonight_hash_ctx = cryptonight_av3_softaes;
            break;

        case XMR_AV4_SOFT_AES_DOUBLE:
            opt_double_hash = true;
            cryptonight_hash_ctx = cryptonight_av4_softaes_double;
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


int scanhash_cryptonight_double(int thr_id, uint32_t *hash, uint8_t *restrict blob, size_t blob_size, uint32_t target, uint32_t max_nonce, unsigned long *restrict hashes_done, struct cryptonight_ctx *restrict ctx) {
    int rc = 0;
    uint32_t *nonceptr0 = (uint32_t*) (((char*) blob) + 39);
    uint32_t *nonceptr1 = (uint32_t*) (((char*) blob) + 39 + blob_size);

    do {
        cryptonight_hash_ctx(blob, blob_size, hash, ctx);
        (*hashes_done) += 2;

        if (unlikely(hash[7] < target)) {
            return rc |= 1;
        }

        if (unlikely(hash[15] < target)) {
            return rc |= 2;
        }

        if (rc) {
            break;
        }

        (*nonceptr0)++;
        (*nonceptr1)++;
    } while (likely(((*nonceptr0) < max_nonce && !work_restart[thr_id].restart)));

    return rc;
}
#endif
