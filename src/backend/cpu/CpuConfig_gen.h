/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#ifndef XMRIG_CPUCONFIG_GEN_H
#define XMRIG_CPUCONFIG_GEN_H


#include "backend/common/Threads.h"
#include "backend/cpu/Cpu.h"
#include "backend/cpu/CpuThreads.h"


namespace xmrig {


static inline size_t generate(const char *key, Threads<CpuThreads> &threads, const Algorithm &algorithm, uint32_t limit)
{
    if (threads.isExist(algorithm) || threads.has(key)) {
        return 0;
    }

    return threads.move(key, Cpu::info()->threads(algorithm, limit));
}


template<Algorithm::Family FAMILY>
static inline size_t generate(Threads<CpuThreads> &, uint32_t) { return 0; }


template<>
size_t inline generate<Algorithm::CN>(Threads<CpuThreads> &threads, uint32_t limit)
{
    size_t count = 0;

    count += generate("cn", threads, Algorithm::CN_1, limit);

    if (!threads.isExist(Algorithm::CN_0)) {
        threads.disable(Algorithm::CN_0);
        ++count;
    }

#   ifdef XMRIG_ALGO_CN_GPU
    count += generate("cn/gpu", threads, Algorithm::CN_GPU, limit);
#   endif

    return count;
}


#ifdef XMRIG_ALGO_CN_LITE
template<>
size_t inline generate<Algorithm::CN_LITE>(Threads<CpuThreads> &threads, uint32_t limit)
{
    size_t count = 0;

    count += generate("cn-lite", threads, Algorithm::CN_LITE_1, limit);

    if (!threads.isExist(Algorithm::CN_LITE_0)) {
        threads.disable(Algorithm::CN_LITE_0);
        ++count;
    }

    return count;
}
#endif


#ifdef XMRIG_ALGO_CN_HEAVY
template<>
size_t inline generate<Algorithm::CN_HEAVY>(Threads<CpuThreads> &threads, uint32_t limit)
{
    return generate("cn-heavy", threads, Algorithm::CN_HEAVY_0, limit);
}
#endif


#ifdef XMRIG_ALGO_CN_PICO
template<>
size_t inline generate<Algorithm::CN_PICO>(Threads<CpuThreads> &threads, uint32_t limit)
{
    return generate("cn-pico", threads, Algorithm::CN_PICO_0, limit);
}
#endif


#ifdef XMRIG_ALGO_RANDOMX
template<>
size_t inline generate<Algorithm::RANDOM_X>(Threads<CpuThreads> &threads, uint32_t limit)
{
    size_t count = 0;
    auto cpuInfo = Cpu::info();
    auto wow     = cpuInfo->threads(Algorithm::RX_WOW, limit);

    if (!threads.isExist(Algorithm::RX_ARQ)) {
        auto arq = cpuInfo->threads(Algorithm::RX_ARQ, limit);
        if (arq == wow) {
            threads.setAlias(Algorithm::RX_ARQ, "rx/wow");
            ++count;
        }
        else {
            count += threads.move("rx/arq", std::move(arq));
        }
    }

    if (!threads.isExist(Algorithm::RX_KEVA)) {
        auto keva = cpuInfo->threads(Algorithm::RX_KEVA, limit);
        if (keva == wow) {
            threads.setAlias(Algorithm::RX_KEVA, "rx/wow");
            ++count;
        }
        else {
            count += threads.move("rx/keva", std::move(keva));
        }
    }

    if (!threads.isExist(Algorithm::RX_WOW)) {
        count += threads.move("rx/wow", std::move(wow));
    }

    if (!threads.isExist(Algorithm::RX_DEFYX)) {
        count += generate("defyx", threads, Algorithm::RX_DEFYX, limit);
    }

    if (!threads.isExist(Algorithm::RX_XLA)) {
        count += generate("panthera", threads, Algorithm::RX_XLA, limit);
    }

    count += generate("rx", threads, Algorithm::RX_0, limit);

    return count;
}
#endif


#ifdef XMRIG_ALGO_ARGON2
template<>
size_t inline generate<Algorithm::ARGON2>(Threads<CpuThreads> &threads, uint32_t limit)
{
    return generate("argon2", threads, Algorithm::AR2_CHUKWA, limit);
}
#endif


#ifdef XMRIG_ALGO_ASTROBWT
template<>
size_t inline generate<Algorithm::ASTROBWT>(Threads<CpuThreads>& threads, uint32_t limit)
{
    return generate("astrobwt", threads, Algorithm::ASTROBWT_DERO, limit);
}
#endif

} /* namespace xmrig */


#endif /* XMRIG_CPUCONFIG_GEN_H */
