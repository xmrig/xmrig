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


#include "crypto/CryptoNight.h"
#include "crypto/CryptoNight_p.h"
#include "crypto/CryptoNight_test.h"
#include "net/Job.h"
#include "net/JobResult.h"
#include "Options.h"


void (*cryptonight_hash_ctx)(const void *input, size_t size, void *output, cryptonight_ctx *ctx) = nullptr;


static void cryptonight_av1_aesni(const void *input, size_t size, void *output, struct cryptonight_ctx *ctx) {
    cryptonight_hash<0x80000, MEMORY, 0x1FFFF0, false>(input, size, output, ctx);
}


static void cryptonight_av2_aesni_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_double_hash<0x80000, MEMORY, 0x1FFFF0, false>(input, size, output, ctx);
}


static void cryptonight_av3_softaes(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_hash<0x80000, MEMORY, 0x1FFFF0, true>(input, size, output, ctx);
}


static void cryptonight_av4_softaes_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_double_hash<0x80000, MEMORY, 0x1FFFF0, true>(input, size, output, ctx);
}


#ifndef XMRIG_NO_AEON
static void cryptonight_lite_av1_aesni(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_hash<0x40000, MEMORY_LITE, 0xFFFF0, false>(input, size, output, ctx);
}


static void cryptonight_lite_av2_aesni_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_double_hash<0x40000, MEMORY_LITE, 0xFFFF0, false>(input, size, output, ctx);
}


static void cryptonight_lite_av3_softaes(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_hash<0x40000, MEMORY_LITE, 0xFFFF0, true>(input, size, output, ctx);
}


static void cryptonight_lite_av4_softaes_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_double_hash<0x40000, MEMORY_LITE, 0xFFFF0, true>(input, size, output, ctx);
}

void (*cryptonight_variations[8])(const void *input, size_t size, void *output, cryptonight_ctx *ctx) = {
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
void (*cryptonight_variations[4])(const void *input, size_t size, void *output, cryptonight_ctx *ctx) = {
            cryptonight_av1_aesni,
            cryptonight_av2_aesni_double,
            cryptonight_av3_softaes,
            cryptonight_av4_softaes_double
        };
#endif


bool CryptoNight::hash(const Job &job, JobResult &result, cryptonight_ctx *ctx)
{
    cryptonight_hash_ctx(job.blob(), job.size(), result.result, ctx);

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


void CryptoNight::hash(const uint8_t *input, size_t size, uint8_t *output, cryptonight_ctx *ctx)
{
    cryptonight_hash_ctx(input, size, output, ctx);
}


bool CryptoNight::selfTest(int algo) {
    if (cryptonight_hash_ctx == nullptr) {
        return false;
    }

    char output[64];

    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) _mm_malloc(sizeof(struct cryptonight_ctx), 16);
    ctx->memory = (uint8_t *) _mm_malloc(MEMORY * 2, 16);

    cryptonight_hash_ctx(test_input, 76, output, ctx);

    _mm_free(ctx->memory);
    _mm_free(ctx);

#   ifndef XMRIG_NO_AEON
    return memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE ? test_output1 : test_output0, (Options::i()->doubleHash() ? 64 : 32)) == 0;
#   else
    return memcmp(output, test_output0, (Options::i()->doubleHash() ? 64 : 32)) == 0;
#   endif
}
