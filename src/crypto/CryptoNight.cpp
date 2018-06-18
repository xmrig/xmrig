/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2018      Sebastian Stolzenberg <https://github.com/sebastianstolzenberg>
 * Copyright 2018      BenDroid    <ben@graef.in>
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

#if defined(XMRIG_ARM)
#   include "crypto/CryptoNight_arm.h"
#else
#   include "crypto/CryptoNight_x86.h"
#endif

#include "crypto/CryptoNight_test.h"

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_aesni(PowVariant powVersion, const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
#   if !defined(XMRIG_ARMv7)
    if (powVersion == PowVariant::POW_V1) {
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else if (powVersion == PowVariant::POW_ALLOY) {
        CryptoNightMultiHash<0x100000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, NUM_HASH_BLOCKS>::hash(input, size, output, ctx);
    } else if (powVersion == PowVariant::POW_XTL) {
        CryptoNightMultiHash<0x80000, POW_XLT_V4_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else if (powVersion == PowVariant::POW_MSR) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else if (powVersion == PowVariant::POW_RTO) {
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, NUM_HASH_BLOCKS>::hashLiteIpbc(input, size, output, ctx);
    }else {
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, NUM_HASH_BLOCKS>::hash(input, size, output, ctx);
    }
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_softaes(PowVariant powVersion, const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
    if (powVersion == PowVariant::POW_V1) {
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else if (powVersion == PowVariant::POW_ALLOY) {
        CryptoNightMultiHash<0x100000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, NUM_HASH_BLOCKS>::hash(input, size, output, ctx);
    } else if (powVersion == PowVariant::POW_XTL) {
        CryptoNightMultiHash<0x80000, POW_XLT_V4_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else if (powVersion == PowVariant::POW_MSR) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else if (powVersion == PowVariant::POW_RTO) {
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, NUM_HASH_BLOCKS>::hashLiteIpbc(input, size, output, ctx);
    } else {
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, NUM_HASH_BLOCKS>::hash(input, size, output, ctx);
    }
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_lite_aesni(PowVariant powVersion, const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
#   if !defined(XMRIG_ARMv7)
    if (powVersion == PowVariant::POW_V1) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else if (powVersion == PowVariant::POW_IPBC) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, NUM_HASH_BLOCKS>::hashLiteIpbc(input, size, output, ctx);
    } else {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, NUM_HASH_BLOCKS>::hash(input, size, output, ctx);
    }
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_lite_softaes(PowVariant powVersion, const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
    if (powVersion == PowVariant::POW_V1) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else if (powVersion == PowVariant::POW_IPBC) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, NUM_HASH_BLOCKS>::hashLiteIpbc(input, size, output, ctx);
    } else {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, NUM_HASH_BLOCKS>::hash(input, size, output, ctx);
    }
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_heavy_aesni(PowVariant powVersion, const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
#   if !defined(XMRIG_ARMv7)
    if (powVersion == PowVariant::POW_XHV) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_HEAVY, 0x3FFFF0, false, NUM_HASH_BLOCKS>::hashHeavyHaven(input, size, output, ctx);
    }
    else {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_HEAVY, 0x3FFFF0, false, NUM_HASH_BLOCKS>::hashHeavy(input, size, output, ctx);
    }
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_heavy_softaes(PowVariant powVersion, const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
    if (powVersion == PowVariant::POW_XHV) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_HEAVY, 0x3FFFF0, true, NUM_HASH_BLOCKS>::hashHeavyHaven(input, size, output, ctx);
    }
    else {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_HEAVY, 0x3FFFF0, true, NUM_HASH_BLOCKS>::hashHeavy(input, size, output, ctx);
    }
}

void (*cryptonight_hash_ctx[MAX_NUM_HASH_BLOCKS])(PowVariant powVersion, const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx);

template <size_t HASH_FACTOR>
void setCryptoNightHashMethods(Options::Algo algo, bool aesni)
{
    switch (algo) {
        case Options::ALGO_CRYPTONIGHT:
            if (aesni) {
                cryptonight_hash_ctx[HASH_FACTOR - 1] = cryptonight_aesni<HASH_FACTOR>;
            } else {
                cryptonight_hash_ctx[HASH_FACTOR - 1] = cryptonight_softaes<HASH_FACTOR>;
            }
            break;

        case Options::ALGO_CRYPTONIGHT_LITE:
            if (aesni) {
                cryptonight_hash_ctx[HASH_FACTOR - 1] = cryptonight_lite_aesni<HASH_FACTOR>;
            } else {
                cryptonight_hash_ctx[HASH_FACTOR - 1] = cryptonight_lite_softaes<HASH_FACTOR>;
            }
            break;

        case Options::ALGO_CRYPTONIGHT_HEAVY:
            if (aesni) {
                cryptonight_hash_ctx[HASH_FACTOR - 1] = cryptonight_heavy_aesni<HASH_FACTOR>;
            } else {
                cryptonight_hash_ctx[HASH_FACTOR - 1] = cryptonight_heavy_softaes<HASH_FACTOR>;
            }
            break;
    }
    // next iteration
    setCryptoNightHashMethods<HASH_FACTOR-1>(algo, aesni);
}

template <>
void setCryptoNightHashMethods<0>(Options::Algo algo, bool aesni)
{
    // template recursion abort
};

bool CryptoNight::init(int algo, bool aesni)
{
    setCryptoNightHashMethods<MAX_NUM_HASH_BLOCKS>(static_cast<Options::Algo>(algo), aesni);
    return selfTest(algo);
}

void CryptoNight::hash(size_t factor, PowVariant powVersion, const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx* ctx)
{
    cryptonight_hash_ctx[factor-1](powVersion, input, size, output, ctx);
}

bool CryptoNight::selfTest(int algo)
{
    if (cryptonight_hash_ctx[0] == nullptr
#if MAX_NUM_HASH_BLOCKS > 1
        || cryptonight_hash_ctx[1] == nullptr
#endif
#if MAX_NUM_HASH_BLOCKS > 2
        || cryptonight_hash_ctx[2] == nullptr
#endif
#if MAX_NUM_HASH_BLOCKS > 3
        || cryptonight_hash_ctx[3] == nullptr
#endif
#if MAX_NUM_HASH_BLOCKS > 4
        || cryptonight_hash_ctx[4] == nullptr
#endif
    ) {
        return false;
    }

    uint8_t output[160];

    auto ctx = (struct cryptonight_ctx*) _mm_malloc(sizeof(struct cryptonight_ctx), 16);
    ctx->memory = (uint8_t *) _mm_malloc(MEMORY * 6, 16);

    bool result = true;
    bool resultLite = true;
    bool resultHeavy = true;

    if (algo == Options::ALGO_CRYPTONIGHT_HEAVY) {
        // cn-heavy

        cryptonight_hash_ctx[0](PowVariant::POW_V0, test_input, 76, output, ctx);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        cryptonight_hash_ctx[1](PowVariant::POW_V0, test_input, 76, output, ctx);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        cryptonight_hash_ctx[2](PowVariant::POW_V0, test_input, 76, output, ctx);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy, 96) == 0;
        #endif

        // cn-heavy haven

        cryptonight_hash_ctx[0](PowVariant::POW_XHV, test_input, 76, output, ctx);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy_haven, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        cryptonight_hash_ctx[1](PowVariant::POW_XHV, test_input, 76, output, ctx);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy_haven, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        cryptonight_hash_ctx[2](PowVariant::POW_XHV, test_input, 76, output, ctx);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy_haven, 96) == 0;
        #endif
    } else if (algo == Options::ALGO_CRYPTONIGHT_LITE) {
        // cn-lite v0

        cryptonight_hash_ctx[0](PowVariant::POW_V0, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_v0_lite, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        cryptonight_hash_ctx[1](PowVariant::POW_V0, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_v0_lite, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        cryptonight_hash_ctx[2](PowVariant::POW_V0, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_v0_lite, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        cryptonight_hash_ctx[3](PowVariant::POW_V0, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_v0_lite, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        cryptonight_hash_ctx[4](PowVariant::POW_V0, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_v0_lite, 160) == 0;
        #endif

        // cn-lite v7 tests

        cryptonight_hash_ctx[0](PowVariant::POW_V1, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output,  test_output_v1_lite, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        cryptonight_hash_ctx[1](PowVariant::POW_V1, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_v1_lite, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        cryptonight_hash_ctx[2](PowVariant::POW_V1, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output,  test_output_v1_lite, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        cryptonight_hash_ctx[3](PowVariant::POW_V1, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output,  test_output_v1_lite, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        cryptonight_hash_ctx[4](PowVariant::POW_V1, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output,  test_output_v1_lite, 160) == 0;
        #endif


        // cn-lite ibpc tests

        cryptonight_hash_ctx[0](PowVariant::POW_IPBC, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_ipbc_lite, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        cryptonight_hash_ctx[1](PowVariant::POW_IPBC, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_ipbc_lite, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        cryptonight_hash_ctx[2](PowVariant::POW_IPBC, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_ipbc_lite, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        cryptonight_hash_ctx[3](PowVariant::POW_IPBC, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_ipbc_lite, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        cryptonight_hash_ctx[4](PowVariant::POW_IPBC, test_input, 76, output, ctx);
        resultLite = resultLite && memcmp(output, test_output_ipbc_lite, 160) == 0;
        #endif

    } else {
        // cn v0

        cryptonight_hash_ctx[0](PowVariant::POW_V0,test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_v0, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        cryptonight_hash_ctx[1](PowVariant::POW_V0, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_v0, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        cryptonight_hash_ctx[2](PowVariant::POW_V0, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_v0, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        cryptonight_hash_ctx[3](PowVariant::POW_V0, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_v0, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        cryptonight_hash_ctx[4](PowVariant::POW_V0, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_v0, 160) == 0;
        #endif

        // cn v7

        cryptonight_hash_ctx[0](PowVariant::POW_V1, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_v1, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        cryptonight_hash_ctx[1](PowVariant::POW_V1, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_v1, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        cryptonight_hash_ctx[2](PowVariant::POW_V1, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_v1, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        cryptonight_hash_ctx[3](PowVariant::POW_V1, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_v1, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        cryptonight_hash_ctx[4](PowVariant::POW_V1, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_v1, 160) == 0;
        #endif

        // cn xtl

        cryptonight_hash_ctx[0](PowVariant::POW_XTL,test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_xtl, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        cryptonight_hash_ctx[1](PowVariant::POW_XTL, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_xtl, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        cryptonight_hash_ctx[2](PowVariant::POW_XTL, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_xtl, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        cryptonight_hash_ctx[3](PowVariant::POW_XTL, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_xtl, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        cryptonight_hash_ctx[4](PowVariant::POW_XTL, test_input, 76, output, ctx);
        result = result && memcmp(output, test_output_xtl, 160) == 0;
        #endif
    }
    _mm_free(ctx->memory);
    _mm_free(ctx);

    return result && resultLite & resultHeavy;
}