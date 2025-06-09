/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CN_ALGO_H
#define XMRIG_CN_ALGO_H


#include <cstddef>
#include <cstdint>


#include "base/crypto/Algorithm.h"


namespace xmrig
{


template<Algorithm::Id ALGO = Algorithm::INVALID>
class CnAlgo
{
public:
    constexpr CnAlgo() {};

#   define ASSERT_CN static_assert(Algorithm::isCN(ALGO), "invalid CRYPTONIGHT algorithm")
    constexpr inline Algorithm::Id base() const  { ASSERT_CN; return Algorithm::base(ALGO); }
    constexpr inline size_t memory() const       { ASSERT_CN; return Algorithm::l3(ALGO); }
    constexpr inline uint32_t iterations() const { ASSERT_CN; return CN_ITER; }
    constexpr inline uint32_t mask() const       { return static_cast<uint32_t>(((memory() - 1) / 16) * 16); }
    constexpr inline uint32_t half_mem() const   { return mask() < memory() / 2; }

    constexpr inline bool isBase1() const        { ASSERT_CN; return Algorithm::base(ALGO) == Algorithm::CN_1; }
    constexpr inline bool isBase2() const        { ASSERT_CN; return Algorithm::base(ALGO) == Algorithm::CN_2; }
    constexpr inline bool is2() const            { return ALGO == Algorithm::CN_2; }
    constexpr inline bool isR() const            { return ALGO == Algorithm::CN_R; }
    constexpr inline bool isHalf() const         { return ALGO == Algorithm::CN_HALF; }
    constexpr inline bool isRTO() const          { return ALGO == Algorithm::CN_RTO; }
    constexpr inline bool isRWZ() const          { return ALGO == Algorithm::CN_RWZ; }
    constexpr inline bool isZLS() const          { return ALGO == Algorithm::CN_ZLS; }
    constexpr inline bool isDouble() const       { return ALGO == Algorithm::CN_DOUBLE; }
    constexpr inline bool isCCX() const          { return ALGO == Algorithm::CN_CCX; }
    constexpr inline bool isHeavy() const        { ASSERT_CN; return Algorithm::family(ALGO) == Algorithm::CN_HEAVY; }
    constexpr inline bool isHeavyTube() const    { return ALGO == Algorithm::CN_HEAVY_TUBE; }
    constexpr inline bool isHeavyXHV() const     { return ALGO == Algorithm::CN_HEAVY_XHV; }
    constexpr inline bool isPico0() const        { return ALGO == Algorithm::CN_PICO_0; }
    constexpr inline bool isPicoTLO() const      { return ALGO == Algorithm::CN_PICO_TLO; }
    constexpr inline bool isUPX2() const         { return ALGO == Algorithm::CN_UPX2; }
    constexpr inline bool isGR0() const          { return ALGO == Algorithm::CN_GR_0; }
    constexpr inline bool isGR1() const          { return ALGO == Algorithm::CN_GR_1; }
    constexpr inline bool isGR2() const          { return ALGO == Algorithm::CN_GR_2; }
    constexpr inline bool isGR3() const          { return ALGO == Algorithm::CN_GR_3; }
    constexpr inline bool isGR4() const          { return ALGO == Algorithm::CN_GR_4; }
    constexpr inline bool isGR5() const          { return ALGO == Algorithm::CN_GR_5; }

    inline static uint32_t iterations(Algorithm::Id algo)
    {
        switch (algo) {
        case Algorithm::CN_0:
        case Algorithm::CN_1:
        case Algorithm::CN_2:
        case Algorithm::CN_R:
        case Algorithm::CN_RTO:
            return CN_ITER;

        case Algorithm::CN_FAST:
        case Algorithm::CN_HALF:
#       ifdef XMRIG_ALGO_CN_LITE
        case Algorithm::CN_LITE_0:
        case Algorithm::CN_LITE_1:
#       endif
#       ifdef XMRIG_ALGO_CN_HEAVY
        case Algorithm::CN_HEAVY_0:
        case Algorithm::CN_HEAVY_TUBE:
        case Algorithm::CN_HEAVY_XHV:
#       endif
        case Algorithm::CN_CCX:
            return CN_ITER / 2;

        case Algorithm::CN_RWZ:
        case Algorithm::CN_ZLS:
            return 0x60000;

        case Algorithm::CN_XAO:
        case Algorithm::CN_DOUBLE:
            return CN_ITER * 2;

#       ifdef XMRIG_ALGO_CN_PICO
        case Algorithm::CN_PICO_0:
        case Algorithm::CN_PICO_TLO:
            return CN_ITER / 8;
#       endif

#       ifdef XMRIG_ALGO_CN_FEMTO
        case Algorithm::CN_UPX2:
            return CN_ITER / 32;
#       endif

        default:
            break;
        }

        return 0;
    }

    inline static uint32_t mask(Algorithm::Id algo)
    {
#       ifdef XMRIG_ALGO_CN_PICO
        if (algo == Algorithm::CN_PICO_0) {
            return 0x1FFF0;
        }
#       endif

#       ifdef XMRIG_ALGO_CN_FEMTO
        if (algo == Algorithm::CN_UPX2) {
            return 0x1FFF0;
        }
#       endif

#       ifdef XMRIG_ALGO_GHOSTRIDER
        if (algo == Algorithm::CN_GR_1) {
            return 0x3FFF0;
        }

        if (algo == Algorithm::CN_GR_5) {
            return 0x1FFF0;
        }
#       endif

        return ((Algorithm::l3(algo) - 1) / 16) * 16;
    }

private:
    constexpr const static uint32_t CN_ITER = 0x80000;
};


template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_FAST>::iterations() const         { return CN_ITER / 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_HALF>::iterations() const         { return CN_ITER / 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_LITE_0>::iterations() const       { return CN_ITER / 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_LITE_1>::iterations() const       { return CN_ITER / 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_HEAVY_0>::iterations() const      { return CN_ITER / 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_HEAVY_TUBE>::iterations() const   { return CN_ITER / 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_HEAVY_XHV>::iterations() const    { return CN_ITER / 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_XAO>::iterations() const          { return CN_ITER * 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_DOUBLE>::iterations() const       { return CN_ITER * 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_RWZ>::iterations() const          { return 0x60000; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_ZLS>::iterations() const          { return 0x60000; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_PICO_0>::iterations() const       { return CN_ITER / 8; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_PICO_TLO>::iterations() const     { return CN_ITER / 8; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_CCX>::iterations() const          { return CN_ITER / 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_UPX2>::iterations() const         { return CN_ITER / 32; }


template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_PICO_0>::mask() const             { return 0x1FFF0; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_UPX2>::mask() const               { return 0x1FFF0; }

#ifdef XMRIG_ALGO_GHOSTRIDER
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GR_0>::iterations() const         { return CN_ITER / 4; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GR_1>::iterations() const         { return CN_ITER / 4; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GR_2>::iterations() const         { return CN_ITER / 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GR_3>::iterations() const         { return CN_ITER / 2; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GR_4>::iterations() const         { return CN_ITER / 8; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GR_5>::iterations() const         { return CN_ITER / 8; }

template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GR_1>::mask() const               { return 0x3FFF0; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GR_5>::mask() const               { return 0x1FFF0; }
#endif


} /* namespace xmrig */


#endif /* XMRIG_CN_ALGO_H */
