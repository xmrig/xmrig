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
static void cryptonight_aesni(const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
#   if !defined(XMRIG_ARMv7)
    if ((reinterpret_cast<const uint8_t*>(input)[0] > 6 && Options::i()->forcePowVersion() == Options::PowVersion::POW_AUTODETECT) ||
            Options::i()->forcePowVersion() == Options::PowVersion::POW_V2) {
        CryptoNightMultiHash<0x80000, MEMORY, 0x1FFFF0, false, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else {
        CryptoNightMultiHash<0x80000, MEMORY, 0x1FFFF0, false, NUM_HASH_BLOCKS>::hash(input, size, output, ctx);
    }
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_softaes(const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
    if ((reinterpret_cast<const uint8_t*>(input)[0] > 6 && Options::i()->forcePowVersion() == Options::PowVersion::POW_AUTODETECT) ||
            Options::i()->forcePowVersion() == Options::PowVersion::POW_V2) {
        CryptoNightMultiHash<0x80000, MEMORY, 0x1FFFF0, true, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else {
        CryptoNightMultiHash<0x80000, MEMORY, 0x1FFFF0, true, NUM_HASH_BLOCKS>::hash(input, size, output, ctx);
    }
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_lite_aesni(const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
#   if !defined(XMRIG_ARMv7)
    if ((reinterpret_cast<const uint8_t*>(input)[0] > 1 && Options::i()->forcePowVersion() == Options::PowVersion::POW_AUTODETECT) ||
            Options::i()->forcePowVersion() == Options::PowVersion::POW_V2) {
        CryptoNightMultiHash<0x40000, MEMORY_LITE, 0xFFFF0, false, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else {
        CryptoNightMultiHash<0x40000, MEMORY_LITE, 0xFFFF0, false, NUM_HASH_BLOCKS>::hash(input, size, output, ctx);
    }
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_lite_softaes(const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
    if ((reinterpret_cast<const uint8_t*>(input)[0] > 1 && Options::i()->forcePowVersion() == Options::PowVersion::POW_AUTODETECT) ||
            Options::i()->forcePowVersion() == Options::PowVersion::POW_V2) {
        CryptoNightMultiHash<0x40000, MEMORY_LITE, 0xFFFF0, true, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, ctx);
    } else {
        CryptoNightMultiHash<0x40000, MEMORY_LITE, 0xFFFF0, true, NUM_HASH_BLOCKS>::hash(input, size, output, ctx);
    }
}


template <size_t NUM_HASH_BLOCKS>
static void cryptonight_heavy_aesni(const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
#   if !defined(XMRIG_ARMv7)
    CryptoNightMultiHash<0x40000, MEMORY_HEAVY, 0x3FFFF0, false, NUM_HASH_BLOCKS>::hashHeavy(input, size, output, ctx);
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_heavy_softaes(const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx) {
    CryptoNightMultiHash<0x40000, MEMORY_HEAVY, 0x3FFFF0, true, NUM_HASH_BLOCKS>::hashHeavy(input, size, output, ctx);
}

void (*cryptonight_hash_ctx[MAX_NUM_HASH_BLOCKS])(const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx *ctx);

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

void CryptoNight::hash(size_t factor, const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx* ctx)
{
    cryptonight_hash_ctx[factor-1](input, size, output, ctx);
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

    bool resultV1Pow = true;
    bool resultV2Pow = true;
    bool resultHeavy = true;

    if (algo == Options::ALGO_CRYPTONIGHT_HEAVY)
    {
        cryptonight_hash_ctx[0](test_input, 76, output, ctx);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        cryptonight_hash_ctx[1](test_input, 76, output, ctx);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        cryptonight_hash_ctx[2](test_input, 76, output, ctx);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy, 96) == 0;
        #endif
    }
    else {
        if (Options::i()->forcePowVersion() == Options::PowVersion::POW_AUTODETECT ||
            Options::i()->forcePowVersion() == Options::PowVersion::POW_V1) {
            cryptonight_hash_ctx[0](test_input, 76, output, ctx);
            resultV1Pow = resultV1Pow &&
                          memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE ? test_output_light : test_output,
                                 32) == 0;

            #if MAX_NUM_HASH_BLOCKS > 1
            cryptonight_hash_ctx[1](test_input, 76, output, ctx);
            resultV1Pow = resultV1Pow &&
                          memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE ? test_output_light : test_output,
                                 64) == 0;
            #endif

            #if MAX_NUM_HASH_BLOCKS > 2
            cryptonight_hash_ctx[2](test_input, 76, output, ctx);
            resultV1Pow = resultV1Pow &&
                          memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE ? test_output_light : test_output,
                                 96) == 0;
            #endif

            #if MAX_NUM_HASH_BLOCKS > 3
            cryptonight_hash_ctx[3](test_input, 76, output, ctx);
            resultV1Pow = resultV1Pow &&
                          memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE ? test_output_light : test_output,
                                 128) == 0;
            #endif

            #if MAX_NUM_HASH_BLOCKS > 4
            cryptonight_hash_ctx[4](test_input, 76, output, ctx);
            resultV1Pow = resultV1Pow &&
                          memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE ? test_output_light : test_output,
                                 160) == 0;
            #endif
        }

        // monero/aeon v2 pow (monero/aeon blockchain version 7)
        if (Options::i()->forcePowVersion() == Options::PowVersion::POW_AUTODETECT ||
            Options::i()->forcePowVersion() == Options::PowVersion::POW_V2) {
            cryptonight_hash_ctx[0](test_input_monero_v2_pow_0, sizeof(test_input_monero_v2_pow_0), output, ctx);
            resultV2Pow = resultV2Pow && memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE
                                                        ? test_output_monero_v2_pow_light[0]
                                                        : test_output_monero_v2_pow[0], 32) == 0;

            #if MAX_NUM_HASH_BLOCKS > 1
            cryptonight_hash_ctx[1](test_input_monero_v2_pow_1, sizeof(test_input_monero_v2_pow_1), output, ctx);
            resultV2Pow = resultV2Pow && memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE
                                                        ? test_output_monero_v2_pow_light[1]
                                                        : test_output_monero_v2_pow[1], 32) == 0;
            #endif

            #if MAX_NUM_HASH_BLOCKS > 2
            cryptonight_hash_ctx[2](test_input_monero_v2_pow_2, sizeof(test_input_monero_v2_pow_2), output, ctx);
            resultV2Pow = resultV2Pow && memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE
                                                        ? test_output_monero_v2_pow_light[2]
                                                        : test_output_monero_v2_pow[2], 32) == 0;
            #endif
        }
    }
    _mm_free(ctx->memory);
    _mm_free(ctx);

    return resultV1Pow && resultV2Pow & resultHeavy;
}