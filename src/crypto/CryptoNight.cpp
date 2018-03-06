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


#include "crypto/CryptoNight.h"


#if defined(XMRIG_ARM)
#   include "crypto/CryptoNight_arm.h"
#else
#   include "crypto/CryptoNight_x86.h"
#endif

#include "crypto/CryptoNight_test.h"
#include "net/Job.h"
#include "net/JobResult.h"
#include "Options.h"



void (*cryptonight_hash_ctx)(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t) = nullptr;


static void cryptonight_av1_aesni(const void *input, size_t size, void *output, struct cryptonight_ctx *ctx, uint8_t version) {
#   if !defined(XMRIG_ARMv7)
    cryptonight_hash<MONERO_ITER, MONERO_MEMORY, MONERO_MASK, false, true>(input, size, output, ctx, version);
#   endif
}


static void cryptonight_av2_aesni_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
#   if !defined(XMRIG_ARMv7)
    cryptonight_double_hash<MONERO_ITER, MONERO_MEMORY, MONERO_MASK, false, true>(input, size, output, ctx, version);
#   endif
}


static void cryptonight_av3_softaes(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
    cryptonight_hash<MONERO_ITER, MONERO_MEMORY, MONERO_MASK, true, true>(input, size, output, ctx, version);
}


static void cryptonight_av4_softaes_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
    cryptonight_double_hash<MONERO_ITER, MONERO_MEMORY, MONERO_MASK, true, true>(input, size, output, ctx, version);
}


#ifndef XMRIG_NO_AEON
static void cryptonight_lite_av1_aesni(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
#   if !defined(XMRIG_ARMv7)
    cryptonight_hash<AEON_ITER, AEON_MEMORY, AEON_MASK, false, false>(input, size, output, ctx, version);
#   endif
}


static void cryptonight_lite_av2_aesni_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
#   if !defined(XMRIG_ARMv7)
    cryptonight_double_hash<AEON_ITER, AEON_MEMORY, AEON_MASK, false, false>(input, size, output, ctx, version);
#   endif
}


static void cryptonight_lite_av3_softaes(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
    cryptonight_hash<AEON_ITER, AEON_MEMORY, AEON_MASK, true, false>(input, size, output, ctx, version);
}


static void cryptonight_lite_av4_softaes_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
    cryptonight_double_hash<AEON_ITER, AEON_MEMORY, AEON_MASK, true, false>(input, size, output, ctx, version);
}

void (*cryptonight_variations[8])(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t) = {
            cryptonight_av1_aesni,
            cryptonight_av2_aesni_double,
            cryptonight_av3_softaes,
            cryptonight_av4_softaes_double,
            cryptonight_lite_av1_aesni,
            cryptonight_lite_av2_aesni_double,
            cryptonight_lite_av3_softaes,
            cryptonight_lite_av4_softaes_double
        };
#else
void (*cryptonight_variations[4])(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t) = {
            cryptonight_av1_aesni,
            cryptonight_av2_aesni_double,
            cryptonight_av3_softaes,
            cryptonight_av4_softaes_double
        };
#endif


bool CryptoNight::hash(const Job &job, JobResult &result, cryptonight_ctx *ctx)
{
    cryptonight_hash_ctx(job.blob(), job.size(), result.result, ctx, job.version());

    return *reinterpret_cast<uint64_t*>(result.result + 24) < job.target();
}


bool CryptoNight::init(int algo, int variant)
{
    if (variant < 1 || variant > 4) {
        return false;
    }

#   ifndef XMRIG_NO_AEON
    const int index = algo == Options::ALGO_CRYPTONIGHT_LITE ? (variant + 3) : (variant - 1);
#   else
    const int index = variant - 1;
#   endif

    cryptonight_hash_ctx = cryptonight_variations[index];

    return selfTest(algo);
}


void CryptoNight::hash(const uint8_t *input, size_t size, uint8_t *output, cryptonight_ctx *ctx, uint8_t version)
{
    cryptonight_hash_ctx(input, size, output, ctx, version);
}


bool CryptoNight::selfTest(int algo) {
    if (cryptonight_hash_ctx == nullptr) {
        return false;
    }

    char output[64];

    struct cryptonight_ctx *ctx = static_cast<cryptonight_ctx *>(_mm_malloc(sizeof(cryptonight_ctx), 16));
    ctx->memory = static_cast<uint8_t *>(_mm_malloc(MONERO_MEMORY * 2, 16));

    cryptonight_hash_ctx(test_input, 76, output, ctx, 0);

    _mm_free(ctx->memory);
    _mm_free(ctx);

#   ifndef XMRIG_NO_AEON
    return memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE ? test_output1 : test_output0, (Options::i()->doubleHash() ? 64 : 32)) == 0;
#   else
    return memcmp(output, test_output0, (Options::i()->doubleHash() ? 64 : 32)) == 0;
#   endif
}
