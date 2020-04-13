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

#ifndef XMRIG_CUDACONFIG_GEN_H
#define XMRIG_CUDACONFIG_GEN_H


#include "backend/common/Threads.h"
#include "backend/cuda/CudaThreads.h"
#include "backend/cuda/wrappers/CudaDevice.h"


#include <algorithm>


namespace xmrig {


static inline size_t generate(const char *key, Threads<CudaThreads> &threads, const Algorithm &algorithm, const std::vector<CudaDevice> &devices)
{
    if (threads.isExist(algorithm) || threads.has(key)) {
        return 0;
    }

    return threads.move(key, CudaThreads(devices, algorithm));
}


template<Algorithm::Family FAMILY>
static inline size_t generate(Threads<CudaThreads> &, const std::vector<CudaDevice> &) { return 0; }


template<>
size_t inline generate<Algorithm::CN>(Threads<CudaThreads> &threads, const std::vector<CudaDevice> &devices)
{
    size_t count = 0;

    count += generate("cn", threads, Algorithm::CN_1, devices);
    count += generate("cn/2", threads, Algorithm::CN_2, devices);

    if (!threads.isExist(Algorithm::CN_0)) {
        threads.disable(Algorithm::CN_0);
        count++;
    }

#   ifdef XMRIG_ALGO_CN_GPU
    count += generate("cn/gpu", threads, Algorithm::CN_GPU, devices);
#   endif

    return count;
}


#ifdef XMRIG_ALGO_CN_LITE
template<>
size_t inline generate<Algorithm::CN_LITE>(Threads<CudaThreads> &threads, const std::vector<CudaDevice> &devices)
{
    size_t count = generate("cn-lite", threads, Algorithm::CN_LITE_1, devices);

    if (!threads.isExist(Algorithm::CN_LITE_0)) {
        threads.disable(Algorithm::CN_LITE_0);
        ++count;
    }

    return count;
}
#endif


#ifdef XMRIG_ALGO_CN_HEAVY
template<>
size_t inline generate<Algorithm::CN_HEAVY>(Threads<CudaThreads> &threads, const std::vector<CudaDevice> &devices)
{
    return generate("cn-heavy", threads, Algorithm::CN_HEAVY_0, devices);
}
#endif


#ifdef XMRIG_ALGO_CN_PICO
template<>
size_t inline generate<Algorithm::CN_PICO>(Threads<CudaThreads> &threads, const std::vector<CudaDevice> &devices)
{
    return generate("cn-pico", threads, Algorithm::CN_PICO_0, devices);
}
#endif


#ifdef XMRIG_ALGO_RANDOMX
template<>
size_t inline generate<Algorithm::RANDOM_X>(Threads<CudaThreads> &threads, const std::vector<CudaDevice> &devices)
{
    size_t count = 0;

    auto rx  = CudaThreads(devices, Algorithm::RX_0);
    auto wow = CudaThreads(devices, Algorithm::RX_WOW);
    auto arq = CudaThreads(devices, Algorithm::RX_ARQ);
    auto kva = CudaThreads(devices, Algorithm::RX_KEVA);

    if (!threads.isExist(Algorithm::RX_WOW) && wow != rx) {
        count += threads.move("rx/wow", std::move(wow));
    }

    if (!threads.isExist(Algorithm::RX_ARQ) && arq != rx) {
        count += threads.move("rx/arq", std::move(arq));
    }

    if (!threads.isExist(Algorithm::RX_KEVA) && kva != rx) {
        count += threads.move("rx/keva", std::move(kva));
    }

    count += threads.move("rx", std::move(rx));

    return count;
}
#endif


#ifdef XMRIG_ALGO_ASTROBWT
template<>
size_t inline generate<Algorithm::ASTROBWT>(Threads<CudaThreads> &threads, const std::vector<CudaDevice> &devices)
{
    return generate("astrobwt", threads, Algorithm::ASTROBWT_DERO, devices);
}
#endif


} /* namespace xmrig */


#endif /* XMRIG_CUDACONFIG_GEN_H */
