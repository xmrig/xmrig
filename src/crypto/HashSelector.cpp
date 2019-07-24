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

#include <Mem.h>
#include <argon2.h>
#include "crypto/HashSelector.h"
#include "crypto/Argon2.h"
#include "crypto/CryptoNight.h"

#if defined(XMRIG_ARM)
#   include "crypto/CryptoNight_arm.h"
#else
#   include "crypto/CryptoNight_x86.h"
#endif

#include "crypto/CryptoNight_test.h"
#include "crypto/Argon2_test.h"

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_aesni(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {
#   if !defined(XMRIG_ARMv7)
    if (variant == PowVariant::POW_V1) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_V1, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_V1, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_V1, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_V2) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_V2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
        if ((asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS <= 2) ||
            (asmOptimization == AsmOptimization::ASM_RYZEN && NUM_HASH_BLOCKS == 1) ||
            (asmOptimization == AsmOptimization::ASM_BULLDOZER && NUM_HASH_BLOCKS == 1)) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_V2, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_V2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_V4) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_V4, NUM_HASH_BLOCKS>::hashPowV4(input, size, output, scratchPad, height);
#else
        if (asmOptimization != AsmOptimization::ASM_OFF && NUM_HASH_BLOCKS <= 2) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_V4, NUM_HASH_BLOCKS>::hashPowV4_asm(input, size, output, scratchPad, height, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_V4, NUM_HASH_BLOCKS>::hashPowV4(input, size, output, scratchPad, height);
        }
#endif
    } else if (variant == PowVariant::POW_WOW) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_WOW, NUM_HASH_BLOCKS>::hashPowV4(input, size, output, scratchPad, height);
#else
        if (asmOptimization != AsmOptimization::ASM_OFF && NUM_HASH_BLOCKS <= 2) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_WOW, NUM_HASH_BLOCKS>::hashPowV4_asm(input, size, output, scratchPad, height, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_WOW, NUM_HASH_BLOCKS>::hashPowV4(input, size, output, scratchPad, height);
        }
#endif
    } else if (variant == PowVariant::POW_ALLOY) {
        CryptoNightMultiHash<0x100000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_ALLOY, NUM_HASH_BLOCKS>::hash(input, size, output, scratchPad);
    } else if (variant == PowVariant::POW_XTL) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_XLT_V4_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_XTL, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_XLT_V4_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_XTL, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_XLT_V4_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_XTL, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_FAST_2) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_FAST_2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
        if ((asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS <= 2) ||
            (asmOptimization == AsmOptimization::ASM_RYZEN && NUM_HASH_BLOCKS == 1) ||
            (asmOptimization == AsmOptimization::ASM_BULLDOZER && NUM_HASH_BLOCKS == 1)) {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_FAST_2, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_FAST_2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_DOUBLE) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x100000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_DOUBLE, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
        if ((asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS <= 2) ||
            (asmOptimization == AsmOptimization::ASM_RYZEN && NUM_HASH_BLOCKS == 1) ||
            (asmOptimization == AsmOptimization::ASM_BULLDOZER && NUM_HASH_BLOCKS == 1)) {
            CryptoNightMultiHash<0x100000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_DOUBLE, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x100000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_DOUBLE, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_ZELERIUS) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x60000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_ZELERIUS, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
        if ((asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS <= 2) ||
            (asmOptimization == AsmOptimization::ASM_RYZEN && NUM_HASH_BLOCKS == 1) ||
            (asmOptimization == AsmOptimization::ASM_BULLDOZER && NUM_HASH_BLOCKS == 1)) {
            CryptoNightMultiHash<0x60000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_ZELERIUS, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x60000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_ZELERIUS, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_RWZ) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x60000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false,POW_RWZ,  NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
        if ((asmOptimization != AsmOptimization::ASM_OFF && NUM_HASH_BLOCKS <= 2)) {
            CryptoNightMultiHash<0x60000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_RWZ, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x60000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_RWZ, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_MSR) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_MSR, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_MSR, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_MSR, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_RTO) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_RTO, NUM_HASH_BLOCKS>::hashLiteTube(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_RTO, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_RTO, NUM_HASH_BLOCKS>::hashLiteTube(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_HOSP) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_HOSP, NUM_HASH_BLOCKS>::hashLiteTube(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_HOSP, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_HOSP, NUM_HASH_BLOCKS>::hashLiteTube(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_XFH) {
        CryptoNightMultiHash<0x20000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_XFH, NUM_HASH_BLOCKS>::hashHeavy(input, size, output, scratchPad);
    } else if (variant == PowVariant::POW_CONCEAL) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_CONCEAL, NUM_HASH_BLOCKS>::hash(input, size, output, scratchPad);
    } else {
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, false, POW_V0, NUM_HASH_BLOCKS>::hash(input, size, output, scratchPad);
    }
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_softaes(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {
    if (variant == PowVariant::POW_V1) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_V1, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_V1, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_V1, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_V2) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_V2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_V2, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_V2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_V4) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_V4, NUM_HASH_BLOCKS>::hashPowV4(input, size, output, scratchPad, height);
#else
        if (asmOptimization != AsmOptimization::ASM_OFF && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_V4, NUM_HASH_BLOCKS>::hashPowV4_asm(input, size, output, scratchPad, height, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_V4, NUM_HASH_BLOCKS>::hashPowV4(input, size, output, scratchPad, height);
        }
#endif
    } else if (variant == PowVariant::POW_WOW) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_WOW,NUM_HASH_BLOCKS>::hashPowV4(input, size, output, scratchPad, height);
#else
        if (asmOptimization != AsmOptimization::ASM_OFF && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_WOW, NUM_HASH_BLOCKS>::hashPowV4_asm(input, size, output, scratchPad, height, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_WOW, NUM_HASH_BLOCKS>::hashPowV4(input, size, output, scratchPad, height);
        }
#endif
    } else if (variant == PowVariant::POW_FAST_2) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_FAST_2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_FAST_2, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_FAST_2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_DOUBLE) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x100000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_DOUBLE, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
        CryptoNightMultiHash<0x100000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_DOUBLE, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
        } else {
        CryptoNightMultiHash<0x100000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_DOUBLE, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_ZELERIUS) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x60000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_ZELERIUS, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x60000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_ZELERIUS, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x60000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_ZELERIUS, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_RWZ) {
        CryptoNightMultiHash<0x60000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_RWZ, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
    } else if (variant == PowVariant::POW_ALLOY) {
        CryptoNightMultiHash<0x100000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_ALLOY, NUM_HASH_BLOCKS>::hash(input, size, output, scratchPad);
    } else if (variant == PowVariant::POW_XTL) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_XLT_V4_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_XTL, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_XLT_V4_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_XTL, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_XLT_V4_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_XTL, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_MSR) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_MSR, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_MSR, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_MSR, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_RTO) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_RTO, NUM_HASH_BLOCKS>::hashLiteTube(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_RTO, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_RTO, NUM_HASH_BLOCKS>::hashLiteTube(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_HOSP) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_HOSP, NUM_HASH_BLOCKS>::hashLiteTube(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_HOSP, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_HOSP, NUM_HASH_BLOCKS>::hashLiteTube(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_XFH) {
        CryptoNightMultiHash<0x20000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_XFH, NUM_HASH_BLOCKS>::hashHeavy(input, size, output, scratchPad);
    } else if (variant == PowVariant::POW_CONCEAL) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_CONCEAL, NUM_HASH_BLOCKS>::hash(input, size, output, scratchPad);
    } else {
        CryptoNightMultiHash<0x80000, POW_DEFAULT_INDEX_SHIFT, MEMORY, 0x1FFFF0, true, POW_V0, NUM_HASH_BLOCKS>::hash(input, size, output, scratchPad);
    }
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_lite_aesni(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {
#   if !defined(XMRIG_ARMv7)
    if (variant == PowVariant::POW_V1) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, POW_V1, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, POW_V1, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, POW_V1, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_TUBE) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, POW_TUBE, NUM_HASH_BLOCKS>::hashLiteTube(input, size, output, scratchPad);
    } else if (variant == PowVariant::POW_UPX) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x20000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, POW_UPX, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x20000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, POW_UPX, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x20000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, POW_UPX, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
        }
#endif
    } else {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, false, POW_V0, NUM_HASH_BLOCKS>::hash(input, size, output, scratchPad);
    }
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_lite_softaes(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {
    if (variant == PowVariant::POW_V1) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, POW_V1, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, POW_V1, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, POW_V1, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
        }
#endif
    } else if (variant == PowVariant::POW_TUBE) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, POW_TUBE, NUM_HASH_BLOCKS>::hashLiteTube(input, size, output, scratchPad);
    } else if (variant == PowVariant::POW_UPX) {
#if defined(XMRIG_ARM)
        CryptoNightMultiHash<0x20000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, POW_UPX, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
#else
        if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
            CryptoNightMultiHash<0x20000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, POW_UPX, NUM_HASH_BLOCKS>::hashPowV2_asm(input, size, output, scratchPad, asmOptimization);
        } else {
            CryptoNightMultiHash<0x20000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, POW_UPX, NUM_HASH_BLOCKS>::hashPowV2(input, size, output, scratchPad);
        }
#endif
    } else {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_LITE, 0xFFFF0, true, POW_V0, NUM_HASH_BLOCKS>::hash(input, size, output, scratchPad);
    }
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_super_lite_aesni(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {

}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_super_lite_softaes(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {

}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_ultra_lite_aesni(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {
#   if !defined(XMRIG_ARMv7)
#if defined(XMRIG_ARM)
    CryptoNightMultiHash<0x10000, POW_DEFAULT_INDEX_SHIFT, MEMORY_ULTRA_LITE, 0x1FFF0, false, POW_TURTLE, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
    if ((asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS <= 2) ||
        (asmOptimization == AsmOptimization::ASM_RYZEN && NUM_HASH_BLOCKS == 1) ||
        (asmOptimization == AsmOptimization::ASM_BULLDOZER && NUM_HASH_BLOCKS == 1)) {
        CryptoNightMultiHash<0x10000, POW_DEFAULT_INDEX_SHIFT, MEMORY_ULTRA_LITE, 0x1FFF0, false, POW_TURTLE, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
    } else {
        CryptoNightMultiHash<0x10000, POW_DEFAULT_INDEX_SHIFT, MEMORY_ULTRA_LITE, 0x1FFF0, false, POW_TURTLE, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
    }
#endif
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_ultra_lite_softaes(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {
#if defined(XMRIG_ARM)
    CryptoNightMultiHash<0x10000, POW_DEFAULT_INDEX_SHIFT, MEMORY_ULTRA_LITE, 0x1FFF0, true, POW_TURTLE, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
    if (asmOptimization == AsmOptimization::ASM_INTEL && NUM_HASH_BLOCKS == 1) {
        CryptoNightMultiHash<0x10000, POW_DEFAULT_INDEX_SHIFT, MEMORY_ULTRA_LITE, 0x1FFF0, true, POW_TURTLE, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
    } else {
        CryptoNightMultiHash<0x10000, POW_DEFAULT_INDEX_SHIFT, MEMORY_ULTRA_LITE, 0x1FFF0, true, POW_TURTLE, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
    }
#endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_extreme_lite_aesni(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {
#   if !defined(XMRIG_ARMv7)
#if defined(XMRIG_ARM)
    CryptoNightMultiHash<0x4000, POW_DEFAULT_INDEX_SHIFT, MEMORY_EXTREME_LITE, 0x1FFF0, false, POW_UPX2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
#else
    if ((asmOptimization != AsmOptimization::ASM_OFF && NUM_HASH_BLOCKS <= 2)) {
        CryptoNightMultiHash<0x4000, POW_DEFAULT_INDEX_SHIFT, MEMORY_EXTREME_LITE, 0x1FFF0, false, POW_UPX2, NUM_HASH_BLOCKS>::hashPowV3_asm(input, size, output, scratchPad, asmOptimization);
    } else {
        CryptoNightMultiHash<0x4000, POW_DEFAULT_INDEX_SHIFT, MEMORY_EXTREME_LITE, 0x1FFF0, false, POW_UPX2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
    }
#endif
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_extreme_lite_softaes(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {
    CryptoNightMultiHash<0x4000, POW_DEFAULT_INDEX_SHIFT, MEMORY_EXTREME_LITE, 0x1FFF0, true, POW_UPX2, NUM_HASH_BLOCKS>::hashPowV3(input, size, output, scratchPad);
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_heavy_aesni(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {
#   if !defined(XMRIG_ARMv7)
    if (variant == PowVariant::POW_XHV) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_HEAVY, 0x3FFFF0, false, POW_XHV, NUM_HASH_BLOCKS>::hashHeavy(input, size, output, scratchPad);
    }
    else if (variant == PowVariant::POW_TUBE) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_HEAVY, 0x3FFFF0, false, POW_TUBE, NUM_HASH_BLOCKS>::hashHeavyTube(input, size, output, scratchPad);
    }
    else {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_HEAVY, 0x3FFFF0, false, POW_V0, NUM_HASH_BLOCKS>::hashHeavy(input, size, output, scratchPad);
    }
#   endif
}

template <size_t NUM_HASH_BLOCKS>
static void cryptonight_heavy_softaes(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad) {
    if (variant == PowVariant::POW_XHV) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_HEAVY, 0x3FFFF0, true, POW_XHV, NUM_HASH_BLOCKS>::hashHeavy(input, size, output, scratchPad);
    }
    else if (variant == PowVariant::POW_TUBE) {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_HEAVY, 0x3FFFF0, true, POW_TUBE, NUM_HASH_BLOCKS>::hashHeavyTube(input, size, output, scratchPad);
    }
    else {
        CryptoNightMultiHash<0x40000, POW_DEFAULT_INDEX_SHIFT, MEMORY_HEAVY, 0x3FFFF0, true, POW_V0, NUM_HASH_BLOCKS>::hashHeavy(input, size, output, scratchPad);
    }
}

template <size_t NUM_HASH_BLOCKS>
static void argon2(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad)
{
    if (variant == PowVariant::POW_ARGON2_CHUKWA || variant == POW_TURTLE) {
        argon2id_hash_raw(3, MEMORY_ARGON2_512/1024, 1, input, size, input, 16, output, 32, scratchPad[0]->memory, MEMORY_ARGON2_512);
    }

    if (variant == PowVariant::POW_ARGON2_WRKZ) {
        argon2id_hash_raw(4, MEMORY_ARGON2_256/1024, 1, input, size, input, 16, output, 32, scratchPad[0]->memory, MEMORY_ARGON2_256);
    }
}

void (*hash_ctx[MAX_NUM_HASH_BLOCKS])(AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad);

template <size_t HASH_FACTOR>
void setHashMethods(Options::Algo algo, bool aesni)
{
    switch (algo) {
        case Options::ALGO_CRYPTONIGHT:
            if (aesni) {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_aesni<HASH_FACTOR>;
            } else {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_softaes<HASH_FACTOR>;
            }
            break;

        case Options::ALGO_CRYPTONIGHT_LITE:
            if (aesni) {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_lite_aesni<HASH_FACTOR>;
            } else {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_lite_softaes<HASH_FACTOR>;
            }
            break;

        case Options::ALGO_CRYPTONIGHT_SUPERLITE:
            if (aesni) {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_super_lite_aesni<HASH_FACTOR>;
            } else {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_super_lite_softaes<HASH_FACTOR>;
            }
            break;

        case Options::ALGO_CRYPTONIGHT_ULTRALITE:
            if (aesni) {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_ultra_lite_aesni<HASH_FACTOR>;
            } else {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_ultra_lite_softaes<HASH_FACTOR>;
            }
            break;

        case Options::ALGO_CRYPTONIGHT_EXTREMELITE:
            if (aesni) {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_extreme_lite_aesni<HASH_FACTOR>;
            } else {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_extreme_lite_softaes<HASH_FACTOR>;
            }
            break;

        case Options::ALGO_CRYPTONIGHT_HEAVY:
            if (aesni) {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_heavy_aesni<HASH_FACTOR>;
            } else {
                hash_ctx[HASH_FACTOR - 1] = cryptonight_heavy_softaes<HASH_FACTOR>;
            }
            break;
        case Options::ALGO_ARGON2_256:
        case Options::ALGO_ARGON2_512:
            hash_ctx[HASH_FACTOR - 1] = argon2<HASH_FACTOR>;
            break;
    }
    // next iteration
    setHashMethods<HASH_FACTOR-1>(algo, aesni);
}

template <>
void setHashMethods<0>(Options::Algo algo, bool aesni)
{
    // template recursion abort
};

bool HashSelector::init(Options::Algo algo, bool aesni)
{
    for (int i = 0; i < 256; ++i)
    {
        const uint64_t index = (((i >> POW_DEFAULT_INDEX_SHIFT) & 6) | (i & 1)) << 1;
        const uint64_t index_xtl = (((i >> POW_XLT_V4_INDEX_SHIFT) & 6) | (i & 1)) << 1;

        variant1_table[i] = i ^ ((0x75310 >> index) & 0x30);
        variant_xtl_table[i] = i ^ ((0x75310 >> index_xtl) & 0x30);
    }

    setHashMethods<MAX_NUM_HASH_BLOCKS>(algo, aesni);

    return Options::i()->skipSelfCheck() ? true : selfCheck(algo);
}

void HashSelector::hash(size_t factor, AsmOptimization asmOptimization, uint64_t height, PowVariant variant, const uint8_t* input, size_t size, uint8_t* output, ScratchPad** scratchPad)
{
    hash_ctx[factor-1](asmOptimization, height, variant, input, size, output, scratchPad);
}

bool HashSelector::selfCheck(Options::Algo algo)
{
    if (hash_ctx[0] == nullptr
    #if MAX_NUM_HASH_BLOCKS > 1
        || hash_ctx[1] == nullptr
    #endif
    #if MAX_NUM_HASH_BLOCKS > 2
        || hash_ctx[2] == nullptr
    #endif
    #if MAX_NUM_HASH_BLOCKS > 3
        || hash_ctx[3] == nullptr
    #endif
    #if MAX_NUM_HASH_BLOCKS > 4
        || hash_ctx[4] == nullptr
    #endif
    )
    {
        return false;
    }

    uint8_t output[160];

    ScratchPad* scratchPads [MAX_NUM_HASH_BLOCKS];

    for (size_t i = 0; i < MAX_NUM_HASH_BLOCKS; ++i) {
        ScratchPad* scratchPad = static_cast<ScratchPad *>(_mm_malloc(sizeof(ScratchPad), 4096));
        scratchPad->memory     = (uint8_t *) _mm_malloc(MEMORY * 6, 16);

        auto* p = reinterpret_cast<uint8_t*>(Mem::allocateExecutableMemory(0x4000));
        scratchPad->generated_code  = reinterpret_cast<cn_mainloop_fun_ms_abi>(p);
        scratchPad->generated_code_double = reinterpret_cast<cn_mainloop_double_fun_ms_abi>(p + 0x2000);

        scratchPad->generated_code_data.variant = PowVariant::LAST_ITEM;
        scratchPad->generated_code_data.height = (uint64_t)(-1);
        scratchPad->generated_code_double_data = scratchPad->generated_code_data;

        scratchPads[i] = scratchPad;
    }

    bool result = true;
    bool resultLite = true;
    bool resultSuperLite = true;
    bool resultUltraLite = true;
    bool resultExtremeLite = true;
    bool resultHeavy = true;
    bool resultArgon2 = true;

    AsmOptimization asmOptimization = Options::i()->asmOptimization();

    if (algo == Options::ALGO_CRYPTONIGHT_HEAVY) {
        // cn-heavy

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy, 96) == 0;
        #endif

        // cn-heavy haven

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_XHV, test_input, 76, output, scratchPads);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy_haven, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_XHV, test_input, 76, output, scratchPads);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy_haven, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 0, PowVariant::POW_XHV, test_input, 76, output, scratchPads);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy_haven, 96) == 0;
        #endif

        // cn-heavy bittube

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_TUBE, test_input, 76, output, scratchPads);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy_tube, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_TUBE, test_input, 76, output, scratchPads);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy_tube, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 0, PowVariant::POW_TUBE, test_input, 76, output, scratchPads);
        resultHeavy = resultHeavy && memcmp(output, test_output_heavy_tube, 96) == 0;
        #endif

    } else if (algo == Options::ALGO_CRYPTONIGHT_LITE) {
        // cn-lite v0

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_v0_lite, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_v0_lite, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_v0_lite, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        hash_ctx[3](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_v0_lite, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        hash_ctx[4](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_v0_lite, 160) == 0;
        #endif

        // cn-lite v7 tests

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_V1, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_v1_lite, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_V1, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_v1_lite, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 0, PowVariant::POW_V1, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output,  test_output_v1_lite, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        hash_ctx[3](asmOptimization, 0, PowVariant::POW_V1, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output,  test_output_v1_lite, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        hash_ctx[4](asmOptimization, 0, PowVariant::POW_V1, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output,  test_output_v1_lite, 160) == 0;
        #endif

        // cn-lite ibpc tests

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_TUBE, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_ipbc_lite, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_TUBE, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_ipbc_lite, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 0, PowVariant::POW_TUBE, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_ipbc_lite, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        hash_ctx[3](asmOptimization, 0, PowVariant::POW_TUBE, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_ipbc_lite, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        hash_ctx[4](asmOptimization, 0, PowVariant::POW_TUBE, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output, test_output_ipbc_lite, 160) == 0;
        #endif

        // cn-lite upx

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_UPX, test_input, 76, output, scratchPads);
        resultLite = resultLite && memcmp(output,  test_output_upx, 32) == 0;

    } else if (algo == Options::ALGO_CRYPTONIGHT_SUPERLITE) {
        return false;
    } else if (algo == Options::ALGO_CRYPTONIGHT_ULTRALITE) {
        // cn ultralite (cnv8 + turtle)

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_TURTLE, test_input, 76, output, scratchPads);
        resultUltraLite = resultUltraLite && memcmp(output,  test_output_turtle, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_TURTLE, test_input, 76, output, scratchPads);
        resultUltraLite = resultUltraLite && memcmp(output,  test_output_turtle, 64) == 0;
        #endif
    } else if (algo == Options::ALGO_CRYPTONIGHT_EXTREMELITE) {
        // cn extremelite (cnv8 + upx2)

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_UPX2, test_input, 76, output, scratchPads);
        resultExtremeLite = resultExtremeLite && memcmp(output,  test_output_upx2, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_UPX2, test_input, 76, output, scratchPads);
        resultExtremeLite = resultExtremeLite && memcmp(output,  test_output_upx2, 64) == 0;
        #endif
    } else if (algo == Options::ALGO_CRYPTONIGHT) {
        // cn v0 aka orignal
        hash_ctx[0](asmOptimization, 0, PowVariant::POW_V0,test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v0, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v0, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v0, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        hash_ctx[3](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v0, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        hash_ctx[4](asmOptimization, 0, PowVariant::POW_V0, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v0, 160) == 0;
        #endif

        // cn v7 aka cnv1

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_V1, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v1, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_V1, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v1, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 0, PowVariant::POW_V1, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v1, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        hash_ctx[3](asmOptimization, 0, PowVariant::POW_V1, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v1, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        hash_ctx[4](asmOptimization, 0, PowVariant::POW_V1, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v1, 160) == 0;
        #endif

        // cnv7 + xtl

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_XTL,test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_xtl, 32) == 0;

        // cnv7 + msr aka cn-fast

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_MSR,test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_msr, 32) == 0;

        // cnv7 + alloy

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_ALLOY,test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_alloy, 32) == 0;

        // cnv7 + hosp/rto

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_HOSP,test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_hosp, 32) == 0;

        // cnv8 aka cnv2

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_V2, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v2, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_V2, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v2, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 0, PowVariant::POW_V2, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v2, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        hash_ctx[3](asmOptimization, 0, PowVariant::POW_V2, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v2, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        hash_ctx[4](asmOptimization, 0, PowVariant::POW_V2, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v2, 160) == 0;
        #endif

        // cn conceal

#if !defined(XMRIG_ARM)

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_CONCEAL, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_conceal, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_CONCEAL, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_conceal, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 0, PowVariant::POW_CONCEAL, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_conceal, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        hash_ctx[3](asmOptimization, 0, PowVariant::POW_CONCEAL, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_conceal, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        hash_ctx[4](asmOptimization, 0, PowVariant::POW_CONCEAL, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_conceal, 160) == 0;
        #endif

#endif

        // cn xfh aka cn-heavy-superfast

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_XFH, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_xfh, 32) == 0;

        // cnv8 + xtl aka cn-fast2

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_FAST_2, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_xtl_v9, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_FAST_2, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_xtl_v9, 64) == 0;
        #endif

        // cnv8 + xcash

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_DOUBLE, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_xcash, 32) == 0;

        // cnv8 + zelerius

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_ZELERIUS, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_zelerius, 32) == 0;

        // cnv8 + rwz

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_RWZ, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_rwz, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 0, PowVariant::POW_RWZ, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_rwz, 64) == 0;
        #endif

        // cnv9 aka cnv4 aka cnv5 aka cnr

        hash_ctx[0](asmOptimization, 10000, PowVariant::POW_V4, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v4, 32) == 0;

        #if MAX_NUM_HASH_BLOCKS > 1
        hash_ctx[1](asmOptimization, 10000, PowVariant::POW_V4, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v4, 64) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 2
        hash_ctx[2](asmOptimization, 10000, PowVariant::POW_V4, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v4, 96) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 3
        hash_ctx[3](asmOptimization, 10000, PowVariant::POW_V4, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v4, 128) == 0;
        #endif

        #if MAX_NUM_HASH_BLOCKS > 4
        hash_ctx[4](asmOptimization, 10000, PowVariant::POW_V4, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v4, 160) == 0;
        #endif

        hash_ctx[0](asmOptimization, 10001, PowVariant::POW_V4, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v4_1, 32) == 0;

        hash_ctx[0](asmOptimization, 10002, PowVariant::POW_V4, test_input, 76, output, scratchPads);
        result = result && memcmp(output, test_output_v4_2, 32) == 0;
    }
    else if (algo == Options::ALGO_ARGON2_256) {
        argon2_select_impl(NULL, NULL);

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_ARGON2_WRKZ, argon2_test_input, 76, output, scratchPads);
        resultArgon2 = resultArgon2 && memcmp(output, argon2_wrkz_test_out, 32) == 0;
    }
    else if (algo == Options::ALGO_ARGON2_512) {
        argon2_select_impl(NULL, NULL);

        hash_ctx[0](asmOptimization, 0, PowVariant::POW_ARGON2_CHUKWA, argon2_test_input, 76, output, scratchPads);
        resultArgon2 = resultArgon2 && memcmp(output, argon2_chukwa_test_out, 32) == 0;
    }

    for (size_t i = 0; i < MAX_NUM_HASH_BLOCKS; ++i) {
        _mm_free(scratchPads[i]->memory);
        _mm_free(scratchPads[i]);
    }

    return result && resultLite && resultSuperLite && resultUltraLite && resultExtremeLite && resultHeavy && resultArgon2;
}
