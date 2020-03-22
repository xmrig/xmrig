/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#ifndef XMRIG_OCLCONFIG_GEN_H
#define XMRIG_OCLCONFIG_GEN_H


#include "backend/common/Threads.h"
#include "backend/opencl/OclThreads.h"


#include <algorithm>


namespace xmrig {


static inline size_t generate(const char *key, Threads<OclThreads> &threads, const Algorithm &algorithm, const std::vector<OclDevice> &devices)
{
    if (threads.isExist(algorithm) || threads.has(key)) {
        return 0;
    }

    return threads.move(key, OclThreads(devices, algorithm));
}


template<Algorithm::Family FAMILY>
static inline size_t generate(Threads<OclThreads> &, const std::vector<OclDevice> &) { return 0; }


template<>
size_t inline generate<Algorithm::CN>(Threads<OclThreads> &threads, const std::vector<OclDevice> &devices)
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
size_t inline generate<Algorithm::CN_LITE>(Threads<OclThreads> &threads, const std::vector<OclDevice> &devices)
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
size_t inline generate<Algorithm::CN_HEAVY>(Threads<OclThreads> &threads, const std::vector<OclDevice> &devices)
{
    return generate("cn-heavy", threads, Algorithm::CN_HEAVY_0, devices);
}
#endif


#ifdef XMRIG_ALGO_CN_PICO
template<>
size_t inline generate<Algorithm::CN_PICO>(Threads<OclThreads> &threads, const std::vector<OclDevice> &devices)
{
    return generate("cn-pico", threads, Algorithm::CN_PICO_0, devices);
}
#endif


#ifdef XMRIG_ALGO_RANDOMX
template<>
size_t inline generate<Algorithm::RANDOM_X>(Threads<OclThreads> &threads, const std::vector<OclDevice> &devices)
{
    size_t count = 0;

    auto rx  = OclThreads(devices, Algorithm::RX_0);
    auto wow = OclThreads(devices, Algorithm::RX_WOW);
    auto arq = OclThreads(devices, Algorithm::RX_ARQ);

    if (!threads.isExist(Algorithm::RX_WOW) && wow != rx) {
        count += threads.move("rx/wow", std::move(wow));
    }

    if (!threads.isExist(Algorithm::RX_ARQ) && arq != rx) {
        count += threads.move("rx/arq", std::move(arq));
    }

    count += threads.move("rx", std::move(rx));

    return count;
}
#endif


#ifdef XMRIG_ALGO_ASTROBWT
template<>
size_t inline generate<Algorithm::ASTROBWT>(Threads<OclThreads>& threads, const std::vector<OclDevice>& devices)
{
    return generate("astrobwt", threads, Algorithm::ASTROBWT_DERO, devices);
}
#endif


static inline std::vector<OclDevice> filterDevices(const std::vector<OclDevice> &devices, const std::vector<uint32_t> &hints)
{
    std::vector<OclDevice> out;
    out.reserve(std::min(devices.size(), hints.size()));

    for (const auto &device  : devices) {
        auto it = std::find(hints.begin(), hints.end(), device.index());
        if (it != hints.end()) {
            out.emplace_back(device);
        }
    }

    return out;
}


} /* namespace xmrig */


#endif /* XMRIG_OCLCONFIG_GEN_H */
