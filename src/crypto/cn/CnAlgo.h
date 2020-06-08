/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

    constexpr inline Algorithm::Id base() const  { static_assert(ALGO > Algorithm::INVALID && ALGO < Algorithm::RX_0, "invalid CRYPTONIGHT algorithm"); return Algorithm::CN_2; }
    constexpr inline bool isHeavy() const        { return memory() == CN_MEMORY * 2; }
    constexpr inline bool isR() const            { return ALGO == Algorithm::CN_R; }
    constexpr inline size_t memory() const       { static_assert(ALGO > Algorithm::INVALID && ALGO < Algorithm::RX_0, "invalid CRYPTONIGHT algorithm"); return CN_MEMORY; }
    constexpr inline uint32_t iterations() const { static_assert(ALGO > Algorithm::INVALID && ALGO < Algorithm::RX_0, "invalid CRYPTONIGHT algorithm"); return CN_ITER; }
    constexpr inline uint32_t mask() const       { return static_cast<uint32_t>(((memory() - 1) / 16) * 16); }

    inline static size_t memory(Algorithm::Id algo)
    {
        Algorithm algorithm(algo);

        return algorithm.isCN() ? algorithm.l3() : 0;
    }

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

#       ifdef XMRIG_ALGO_CN_GPU
        case Algorithm::CN_GPU:
            return 0xC000;
#       endif

#       ifdef XMRIG_ALGO_CN_PICO
        case Algorithm::CN_PICO_0:
        case Algorithm::CN_PICO_TLO:
            return CN_ITER / 8;
#       endif

        default:
            break;
        }

        return 0;
    }

    inline static uint32_t mask(Algorithm::Id algo)
    {
#       ifdef XMRIG_ALGO_CN_GPU
        if (algo == Algorithm::CN_GPU) {
            return 0x1FFFC0;
        }
#       endif

#       ifdef XMRIG_ALGO_CN_PICO
        if (algo == Algorithm::CN_PICO_0) {
            return 0x1FFF0;
        }
#       endif

        return ((memory(algo) - 1) / 16) * 16;
    }

    inline static Algorithm::Id base(Algorithm::Id algo)
    {
        switch (algo) {
        case Algorithm::CN_0:
        case Algorithm::CN_XAO:
#       ifdef XMRIG_ALGO_CN_LITE
        case Algorithm::CN_LITE_0:
#       endif
#       ifdef XMRIG_ALGO_CN_HEAVY
        case Algorithm::CN_HEAVY_0:
        case Algorithm::CN_HEAVY_XHV:
#       endif
        case Algorithm::CN_CCX:
            return Algorithm::CN_0;

        case Algorithm::CN_1:
        case Algorithm::CN_FAST:
        case Algorithm::CN_RTO:
#       ifdef XMRIG_ALGO_CN_LITE
        case Algorithm::CN_LITE_1:
#       endif
#       ifdef XMRIG_ALGO_CN_HEAVY
        case Algorithm::CN_HEAVY_TUBE:
            return Algorithm::CN_1;
#       endif

        case Algorithm::CN_2:
        case Algorithm::CN_R:
        case Algorithm::CN_HALF:
        case Algorithm::CN_RWZ:
        case Algorithm::CN_ZLS:
        case Algorithm::CN_DOUBLE:
#       ifdef XMRIG_ALGO_CN_PICO
        case Algorithm::CN_PICO_0:
        case Algorithm::CN_PICO_TLO:
#       endif
            return Algorithm::CN_2;

#       ifdef XMRIG_ALGO_CN_GPU
        case Algorithm::CN_GPU:
            return Algorithm::CN_GPU;
#       endif

        default:
            break;
        }

        return Algorithm::INVALID;
    }

private:
    constexpr const static size_t   CN_MEMORY = 0x200000;
    constexpr const static uint32_t CN_ITER   = 0x80000;
};


template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_0>::base() const             { return Algorithm::CN_0; }
template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_XAO>::base() const           { return Algorithm::CN_0; }
template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_LITE_0>::base() const        { return Algorithm::CN_0; }
template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_HEAVY_0>::base() const       { return Algorithm::CN_0; }
template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_HEAVY_XHV>::base() const     { return Algorithm::CN_0; }
template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_CCX>::base() const           { return Algorithm::CN_0; }
template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_1>::base() const             { return Algorithm::CN_1; }
template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_FAST>::base() const          { return Algorithm::CN_1; }
template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_RTO>::base() const           { return Algorithm::CN_1; }
template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_LITE_1>::base() const        { return Algorithm::CN_1; }
template<> constexpr inline Algorithm::Id CnAlgo<Algorithm::CN_HEAVY_TUBE>::base() const    { return Algorithm::CN_1; }


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
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GPU>::iterations() const          { return 0xC000; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_PICO_0>::iterations() const       { return CN_ITER / 8; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_PICO_TLO>::iterations() const     { return CN_ITER / 8; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_CCX>::iterations() const          { return CN_ITER / 2; }


template<> constexpr inline size_t CnAlgo<Algorithm::CN_LITE_0>::memory() const             { return CN_MEMORY / 2; }
template<> constexpr inline size_t CnAlgo<Algorithm::CN_LITE_1>::memory() const             { return CN_MEMORY / 2; }
template<> constexpr inline size_t CnAlgo<Algorithm::CN_HEAVY_0>::memory() const            { return CN_MEMORY * 2; }
template<> constexpr inline size_t CnAlgo<Algorithm::CN_HEAVY_TUBE>::memory() const         { return CN_MEMORY * 2; }
template<> constexpr inline size_t CnAlgo<Algorithm::CN_HEAVY_XHV>::memory() const          { return CN_MEMORY * 2; }
template<> constexpr inline size_t CnAlgo<Algorithm::CN_PICO_0>::memory() const             { return CN_MEMORY / 8; }
template<> constexpr inline size_t CnAlgo<Algorithm::CN_PICO_TLO>::memory() const           { return CN_MEMORY / 8; }


template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GPU>::mask() const                { return 0x1FFFC0; }
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_PICO_0>::mask() const             { return 0x1FFF0; }


} /* namespace xmrig */


#endif /* XMRIG_CN_ALGO_H */
