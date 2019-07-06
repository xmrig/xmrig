/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <stddef.h>
#include <stdint.h>


#include "crypto/common/Algorithm.h"


namespace xmrig
{


template<Algorithm::Id ALGO = Algorithm::INVALID>
class CnAlgo
{
public:
    constexpr inline CnAlgo()
    {
        static_assert(ALGO != Algorithm::INVALID && m_memory[ALGO] > 0,                 "invalid CRYPTONIGHT algorithm");
        static_assert(sizeof(m_memory)     / sizeof(m_memory)[0]     == Algorithm::MAX, "memory table size mismatch");
        static_assert(sizeof(m_iterations) / sizeof(m_iterations)[0] == Algorithm::MAX, "iterations table size mismatch");
        static_assert(sizeof(m_base)       / sizeof(m_base)[0]       == Algorithm::MAX, "iterations table size mismatch");
    }

    constexpr inline Algorithm::Id base() const  { return m_base[ALGO]; }
    constexpr inline bool isHeavy() const        { return memory() == CN_MEMORY * 2; }
    constexpr inline bool isR() const            { return ALGO == Algorithm::CN_R || ALGO == Algorithm::CN_WOW; }
    constexpr inline size_t memory() const       { return m_memory[ALGO]; }
    constexpr inline uint32_t iterations() const { return m_iterations[ALGO]; }
    constexpr inline uint32_t mask() const       { return ((memory() - 1) / 16) * 16; }

    inline static size_t memory(Algorithm::Id algo)
    {
        switch (Algorithm::family(algo)) {
        case Algorithm::CN:
            return CN_MEMORY;

        case Algorithm::CN_LITE:
            return CN_MEMORY / 2;

        case Algorithm::CN_HEAVY:
            return CN_MEMORY * 2;

        case Algorithm::CN_PICO:
            return CN_MEMORY / 8;

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

private:
    constexpr const static size_t   CN_MEMORY = 0x200000;
    constexpr const static uint32_t CN_ITER   = 0x80000;

    constexpr const static size_t m_memory[] = {
        CN_MEMORY, // CN_0
        CN_MEMORY, // CN_1
        CN_MEMORY, // CN_2
        CN_MEMORY, // CN_R
        CN_MEMORY, // CN_WOW
        CN_MEMORY, // CN_FAST
        CN_MEMORY, // CN_HALF
        CN_MEMORY, // CN_XAO
        CN_MEMORY, // CN_RTO
        CN_MEMORY, // CN_RWZ
        CN_MEMORY, // CN_ZLS
        CN_MEMORY, // CN_DOUBLE
#       ifdef XMRIG_ALGO_CN_GPU
        CN_MEMORY, // CN_GPU
#       endif
#       ifdef XMRIG_ALGO_CN_LITE
        CN_MEMORY / 2, // CN_LITE_0
        CN_MEMORY / 2, // CN_LITE_1
#       endif
#       ifdef XMRIG_ALGO_CN_HEAVY
        CN_MEMORY * 2, // CN_HEAVY_0
        CN_MEMORY * 2, // CN_HEAVY_TUBE
        CN_MEMORY * 2, // CN_HEAVY_XHV
#       endif
#       ifdef XMRIG_ALGO_CN_PICO
        CN_MEMORY / 8, // CN_PICO_0
#       endif
#       ifdef XMRIG_ALGO_RANDOMX
        0,             // RX_0
        0,             // RX_WOW
        0,             // RX_LOKI
#       endif
    };

    constexpr const static uint32_t m_iterations[] = {
        CN_ITER,     // CN_0
        CN_ITER,     // CN_1
        CN_ITER,     // CN_2
        CN_ITER,     // CN_R
        CN_ITER,     // CN_WOW
        CN_ITER / 2, // CN_FAST
        CN_ITER / 2, // CN_HALF
        CN_ITER * 2, // CN_XAO
        CN_ITER,     // CN_RTO
        0x60000,     // CN_RWZ
        0x60000,     // CN_ZLS
        CN_ITER * 2, // CN_DOUBLE
#       ifdef XMRIG_ALGO_CN_GPU
        0xC000,      // CN_GPU
#       endif
#       ifdef XMRIG_ALGO_CN_LITE
        CN_ITER / 2, // CN_LITE_0
        CN_ITER / 2, // CN_LITE_1
#       endif
#       ifdef XMRIG_ALGO_CN_HEAVY
        CN_ITER / 2, // CN_HEAVY_0
        CN_ITER / 2, // CN_HEAVY_TUBE
        CN_ITER / 2, // CN_HEAVY_XHV
#       endif
#       ifdef XMRIG_ALGO_CN_PICO
        CN_ITER / 8, // CN_PICO_0
#       endif
#       ifdef XMRIG_ALGO_RANDOMX
        0,             // RX_0
        0,             // RX_WOW
        0,             // RX_LOKI
#       endif
    };

    constexpr const static Algorithm::Id m_base[] = {
        Algorithm::CN_0,   // CN_0
        Algorithm::CN_1,   // CN_1
        Algorithm::CN_2,   // CN_2
        Algorithm::CN_2,   // CN_R
        Algorithm::CN_2,   // CN_WOW
        Algorithm::CN_1,   // CN_FAST
        Algorithm::CN_2,   // CN_HALF
        Algorithm::CN_0,   // CN_XAO
        Algorithm::CN_1,   // CN_RTO
        Algorithm::CN_2,   // CN_RWZ
        Algorithm::CN_2,   // CN_ZLS
        Algorithm::CN_2,   // CN_DOUBLE
#       ifdef XMRIG_ALGO_CN_GPU
        Algorithm::CN_GPU, // CN_GPU
#       endif
#       ifdef XMRIG_ALGO_CN_LITE
        Algorithm::CN_0,   // CN_LITE_0
        Algorithm::CN_1,   // CN_LITE_1
#       endif
#       ifdef XMRIG_ALGO_CN_HEAVY
        Algorithm::CN_0,   // CN_HEAVY_0
        Algorithm::CN_1,   // CN_HEAVY_TUBE
        Algorithm::CN_0,   // CN_HEAVY_XHV
#       endif
#       ifdef XMRIG_ALGO_CN_PICO
        Algorithm::CN_2,    // CN_PICO_0,
#       endif
#       ifdef XMRIG_ALGO_RANDOMX
        Algorithm::INVALID, // RX_0
        Algorithm::INVALID, // RX_WOW
        Algorithm::INVALID, // RX_LOKI
#       endif
    };
};


#ifdef XMRIG_ALGO_CN_GPU
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_GPU>::mask() const { return 0x1FFFC0; }
#endif

#ifdef XMRIG_ALGO_CN_PICO
template<> constexpr inline uint32_t CnAlgo<Algorithm::CN_PICO_0>::mask() const { return 0x1FFF0; }
#endif


} /* namespace xmrig */


#endif /* XMRIG_CN_ALGO_H */
