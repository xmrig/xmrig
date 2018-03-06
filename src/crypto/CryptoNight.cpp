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

// VARIANT ALTERATIONS
#define VARIANT1_CHECK() \
    if (MONERO && version > 6 && size < 43) \
    { \
        return false; \
    }

#define VARIANT1_INIT(part) \
    const uint64_t tweak1_2_##part =\
        (MONERO && version > 6) ? \
            (*reinterpret_cast<const uint64_t*>(reinterpret_cast<const uint8_t*>(input) + 35 + part * size) ^ \
            *(reinterpret_cast<const uint64_t*>(ctx->state##part) + 24)) : 0;

#define VARIANT1_1(p) \
    do if(version > 6) \
    { \
        const uint8_t tmp = reinterpret_cast<const uint8_t*>(p)[11]; \
        static const uint32_t table = 0x75310; \
        const uint8_t index = (((tmp >> 3) & 6) | (tmp & 1)) << 1; \
        ((uint8_t*)(p))[11] = tmp ^ ((table >> index) & 0x30); \
  } while(0)

#define VARIANT1_2(p, part) \
    do if(MONERO && version > 6) { \
        (p) ^= tweak1_2_##part ; \
    } while(0)


#if defined(XMRIG_ARM)
#   include "crypto/CryptoNight_arm.h"
#else
#   include "crypto/CryptoNight_x86.h"
#endif

#include "crypto/CryptoNight_test.h"
#include "net/Job.h"
#include "net/JobResult.h"
#include "Options.h"



bool (*cryptonight_hash_ctx)(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t) = nullptr;


static bool cryptonight_av1_aesni(const void *input, size_t size, void *output, struct cryptonight_ctx *ctx, uint8_t version) {
#   if !defined(XMRIG_ARMv7)
    return cryptonight_hash<0x80000, MEMORY, 0x1FFFF0, false, true>(input, size, output, ctx, version);
#   else
    return false;
#   endif
}


static bool cryptonight_av2_aesni_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
#   if !defined(XMRIG_ARMv7)
    return cryptonight_double_hash<0x80000, MEMORY, 0x1FFFF0, false, true>(input, size, output, ctx, version);
#   else
    return false;
#   endif
}


static bool cryptonight_av3_softaes(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
    return cryptonight_hash<0x80000, MEMORY, 0x1FFFF0, true, true>(input, size, output, ctx, version);
}


static bool cryptonight_av4_softaes_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
    return cryptonight_double_hash<0x80000, MEMORY, 0x1FFFF0, true, true>(input, size, output, ctx, version);
}


#ifndef XMRIG_NO_AEON
static bool cryptonight_lite_av1_aesni(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
#   if !defined(XMRIG_ARMv7)
    return cryptonight_hash<0x40000, MEMORY_LITE, 0xFFFF0, false, false>(input, size, output, ctx, version);
#   else
    return false;
#   endif
}


static bool cryptonight_lite_av2_aesni_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
#   if !defined(XMRIG_ARMv7)
    return cryptonight_double_hash<0x40000, MEMORY_LITE, 0xFFFF0, false, false>(input, size, output, ctx, version);
#   else
    return false;
#   endif
}


static bool cryptonight_lite_av3_softaes(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
    return cryptonight_hash<0x40000, MEMORY_LITE, 0xFFFF0, true, false>(input, size, output, ctx, version);
}


static bool cryptonight_lite_av4_softaes_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t version) {
    return cryptonight_double_hash<0x40000, MEMORY_LITE, 0xFFFF0, true, false>(input, size, output, ctx, version);
}

bool (*cryptonight_variations[8])(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t) = {
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
bool (*cryptonight_variations[4])(const void *input, size_t size, void *output, cryptonight_ctx *ctx, uint8_t) = {
            cryptonight_av1_aesni,
            cryptonight_av2_aesni_double,
            cryptonight_av3_softaes,
            cryptonight_av4_softaes_double
        };
#endif


bool CryptoNight::hash(const Job &job, JobResult &result, cryptonight_ctx *ctx, uint8_t version)
{
    const bool rc = cryptonight_hash_ctx(job.blob(), job.size(), result.result, ctx, version);
    return rc && *reinterpret_cast<uint64_t*>(result.result + 24) < job.target();
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


bool CryptoNight::hash(const uint8_t *input, size_t size, uint8_t *output, cryptonight_ctx *ctx, uint8_t version)
{
    return cryptonight_hash_ctx(input, size, output, ctx, version);
}


bool CryptoNight::selfTest(int algo) {
    if (cryptonight_hash_ctx == nullptr) {
        return false;
    }

    char output[64];

    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) _mm_malloc(sizeof(struct cryptonight_ctx), 16);
    ctx->memory = (uint8_t *) _mm_malloc(MEMORY * 2, 16);

    const bool rc = cryptonight_hash_ctx(test_input, 76, output, ctx, 0);

    _mm_free(ctx->memory);
    _mm_free(ctx);

#   ifndef XMRIG_NO_AEON
    return rc && memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE ? test_output1 : test_output0, (Options::i()->doubleHash() ? 64 : 32)) == 0;
#   else
    return rc && memcmp(output, test_output0, (Options::i()->doubleHash() ? 64 : 32)) == 0;
#   endif
}
