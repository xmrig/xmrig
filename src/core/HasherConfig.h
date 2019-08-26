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

#ifndef XMRIG_HASHERCONFIG_H
#define XMRIG_HASHERCONFIG_H


#include "common/xmrig.h"
#include "crypto/argon2_hasher/common/common.h"

namespace xmrig {

struct GPUFilter {
    GPUFilter(std::string engine, std::string filter) : engine(engine), filter(filter) {}
    std::string engine;
    std::string filter;
};

class DLLEXPORT HasherConfig
{
public:
    HasherConfig(Algo algorithm,
                 Variant variant,
                 int priority,
                 int cpuThreads,
                 int64_t cpuAffinity,
                 std::string cpuOptimization,
                 std::vector<double> &gpuIntensity,
                 std::vector<GPUFilter> &gpuFilter);

    HasherConfig *clone(int index, std::string hasherType);

    inline size_t index() const         { return m_index; }
    inline std::string type() const     { return m_type; }
    inline Algo algorithm() const       { return m_algorithm; }
    inline Variant variant() const      { return m_variant; }
    inline int priority() const         { return m_priority; }
    inline int cpuThreads() const    { return m_cpuThreads; }
    inline std::string cpuOptimization() const { return m_cpuOptimization; }
    inline std::vector<GPUFilter> &gpuFilter() { return m_gpuFilter; }

    double getAverageGPUIntensity();
    double getGPUIntensity(int cardIndex);
    int64_t getCPUAffinity(int cpuIndex);

    inline void addGPUCardsCount(int count) { m_gpuCardsCount += count; }
    inline int getGPUCardsCount() { return m_gpuCardsCount; }

private:
    HasherConfig(int index,
                 std::string type,
                 Algo algorithm,
                 Variant variant,
                 int priority,
                 int cpuThreads,
                 int64_t cpuAffinity,
                 std::string cpuOptimization,
                 std::vector<double> &gpuIntensity,
                 std::vector<GPUFilter> &gpuFilter);

    const size_t m_index;
    const std::string m_type;
    const Algo m_algorithm;
    const Variant m_variant;
    const int m_priority;
    const int m_cpuThreads;
    const int64_t m_cpuAffinity;
    const std::string m_cpuOptimization;
    std::vector<double> m_gpuIntensity;
    std::vector<GPUFilter> m_gpuFilter;

    static int m_gpuCardsCount;
};

} /* namespace xmrig */

#endif /*XMRIG_HASHERCONFIG_H*/
