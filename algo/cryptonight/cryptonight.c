/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "crypto/c_blake256.h"
#include "crypto/c_groestl.h"
#include "crypto/c_jh.h"
#include "crypto/c_skein.h"
#include "cryptonight.h"
#include "cryptonight_test.h"
#include "options.h"


void cryptonight_av1_aesni(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx, uint8_t version);
void cryptonight_av2_aesni_double(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx, uint8_t version);
void cryptonight_av3_softaes(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx, uint8_t version);
void cryptonight_av4_softaes_double(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx, uint8_t version);

#ifndef XMRIG_NO_AEON
void cryptonight_lite_av1_aesni(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx, uint8_t);
void cryptonight_lite_av2_aesni_double(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx, uint8_t);
void cryptonight_lite_av3_softaes(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx, uint8_t);
void cryptonight_lite_av4_softaes_double(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx, uint8_t);
#endif

void (*cryptonight_hash_ctx)(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx, uint8_t version) = NULL;


static bool self_test() {
    if (cryptonight_hash_ctx == NULL) {
        return false;
    }

    char output[64];

    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) _mm_malloc(sizeof(struct cryptonight_ctx), 16);
    ctx->memory = (uint8_t *) _mm_malloc(MEMORY * 2, 16);

    cryptonight_hash_ctx(test_input, 76, output, ctx, 0);

#   ifndef XMRIG_NO_AEON
    bool rc = memcmp(output, opt_algo == ALGO_CRYPTONIGHT_LITE ? test_output1 : test_output0, (opt_double_hash ? 64 : 32)) == 0;
#   else
    bool rc = memcmp(output, test_output0, opt_double_hash ? 64 : 32)) == 0;
#   endif

    if (rc && opt_algo == ALGO_CRYPTONIGHT) {
        cryptonight_hash_ctx(test_input, 76, output, ctx, 7);

        rc = memcmp(output, test_output2, (opt_double_hash ? 64 : 32)) == 0;
    }

    _mm_free(ctx->memory);
    _mm_free(ctx);

    return rc;
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
        cryptonight_hash_ctx(blob, blob_size, hash, ctx, ((uint8_t*) blob)[0]);
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
        cryptonight_hash_ctx(blob, blob_size, hash, ctx, ((uint8_t*) blob)[0]);
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
