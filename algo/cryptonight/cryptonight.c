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


#include <assert.h>
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

#include "utils/applog.h"


void cryptonight_av1_v0(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av1_v1(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av1_v2(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av2_v0(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av2_v1(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av2_v2(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av3_v0(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av3_v1(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av3_v2(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av4_v0(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av4_v1(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_av4_v2(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);


#ifndef XMRIG_NO_AEON
void cryptonight_lite_av1_v0(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_lite_av1_v1(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_lite_av2_v0(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_lite_av2_v1(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_lite_av3_v0(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_lite_av3_v1(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_lite_av4_v0(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
void cryptonight_lite_av4_v1(const uint8_t *input, size_t size, uint8_t *output, struct cryptonight_ctx **ctx);
#endif

void (*cryptonight_hash_ctx)(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx, uint8_t version) = NULL;


static inline bool verify(enum Variant variant, uint8_t *output, struct cryptonight_ctx **ctx, const uint8_t *referenceValue)
{
    cn_hash_fun func = cryptonight_hash_fn(opt_algo, opt_av, variant);
    if (func == NULL) {
        return false;
    }

    func(test_input, 76, output, ctx);

    return memcmp(output, referenceValue, opt_double_hash ? 64 : 32) == 0;
}


static bool self_test() {
    struct cryptonight_ctx *ctx[2];
    uint8_t output[64];

    const size_t count = opt_double_hash ? 2 : 1;
    const size_t size  = opt_algo == ALGO_CRYPTONIGHT ? MEMORY : MEMORY_LITE;
    bool result = false;

    for (int i = 0; i < count; ++i) {
        ctx[i]         = _mm_malloc(sizeof(struct cryptonight_ctx), 16);
        ctx[i]->memory = _mm_malloc(size, 16);
    }

    if (opt_algo == ALGO_CRYPTONIGHT) {
        result = verify(VARIANT_0, output, ctx, test_output_v0) &&
                 verify(VARIANT_1, output, ctx, test_output_v1) &&
                 verify(VARIANT_0, output, ctx, test_output_v0);
    }
    else {
        result = verify(VARIANT_0, output, ctx, test_output_v0_lite) &&
                 verify(VARIANT_1, output, ctx, test_output_v1_lite);
    }


    for (int i = 0; i < count; ++i) {
        _mm_free(ctx[i]->memory);
        _mm_free(ctx[i]);
    }

    return result;
}


cn_hash_fun cryptonight_hash_fn(enum Algo algorithm, enum AlgoVariant av, enum Variant variant)
{
    assert(av > AV_AUTO && av < AV_MAX);
    assert(variant > VARIANT_AUTO && variant < VARIANT_MAX);

    static const cn_hash_fun func_table[VARIANT_MAX * 4 * 2] = {
        cryptonight_av1_v0,
        cryptonight_av2_v0,
        cryptonight_av3_v0,
        cryptonight_av4_v0,
        cryptonight_av1_v1,
        cryptonight_av2_v1,
        cryptonight_av3_v1,
        cryptonight_av4_v1,
        cryptonight_av1_v2,
        cryptonight_av2_v2,
        cryptonight_av3_v2,
        cryptonight_av4_v2,

#       ifndef XMRIG_NO_AEON
        cryptonight_lite_av1_v0,
        cryptonight_lite_av2_v0,
        cryptonight_lite_av3_v0,
        cryptonight_lite_av4_v0,
        cryptonight_lite_av1_v1,
        cryptonight_lite_av2_v1,
        cryptonight_lite_av3_v1,
        cryptonight_lite_av4_v1,
        NULL,
        NULL,
        NULL,
        NULL
#       endif
    };

    const size_t index = VARIANT_MAX * 4 * algorithm + 4 * variant + av - 1;

#   ifndef NDEBUG
    cn_hash_fun func = func_table[index];

    assert(index < sizeof(func_table) / sizeof(func_table[0]));
    assert(func != NULL);

    return func;
#   else
    return func_table[index];
#   endif
}


bool cryptonight_init(int av)
{
    opt_double_hash = av == AV_DOUBLE || av == AV_DOUBLE_SOFT;

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


static inline enum Variant cryptonight_variant(uint8_t version)
{
    if (opt_variant != VARIANT_AUTO) {
        return opt_variant;
    }

    if (opt_algo == ALGO_CRYPTONIGHT_LITE) {
        return VARIANT_1;
    }

    if (version >= 8) {
        return VARIANT_2;
    }

    return version == 7 ? VARIANT_1 : VARIANT_0;
}


#ifndef BUILD_TEST
int scanhash_cryptonight(int thr_id, uint32_t *hash, const uint8_t *restrict blob, size_t blob_size, uint32_t target, uint32_t max_nonce, unsigned long *restrict hashes_done, struct cryptonight_ctx **restrict ctx) {
    uint32_t *nonceptr   = (uint32_t*) (((char*) blob) + 39);
    enum Variant variant = cryptonight_variant(blob[0]);

    do {
        cryptonight_hash_fn(opt_algo, opt_av, variant)(blob, blob_size, (uint8_t *) hash, ctx);

        (*hashes_done)++;

        if (unlikely(hash[7] < target)) {
            return 1;
        }

        (*nonceptr)++;
    } while (likely(((*nonceptr) < max_nonce && !work_restart[thr_id].restart)));

    return 0;
}


int scanhash_cryptonight_double(int thr_id, uint32_t *hash, const uint8_t *restrict blob, size_t blob_size, uint32_t target, uint32_t max_nonce, unsigned long *restrict hashes_done, struct cryptonight_ctx **restrict ctx) {
    int rc               = 0;
    uint32_t *nonceptr0  = (uint32_t*) (((char*) blob) + 39);
    uint32_t *nonceptr1  = (uint32_t*) (((char*) blob) + 39 + blob_size);
    enum Variant variant = cryptonight_variant(blob[0]);

    do {
        cryptonight_hash_fn(opt_algo, opt_av, variant)(blob, blob_size, (uint8_t *) hash, ctx);
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
