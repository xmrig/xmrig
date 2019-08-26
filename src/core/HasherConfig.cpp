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

#include <assert.h>
#include <string>
#include <vector>
#include <cstdint>

#include "crypto/argon2_hasher/common/DLLExport.h"

#include "HasherConfig.h"

int xmrig::HasherConfig::m_gpuCardsCount = 0;

xmrig::HasherConfig::HasherConfig(xmrig::Algo algorithm, xmrig::Variant variant, int priority, int cpuThreads,
                                  int64_t cpuAffinity, std::string cpuOptimization,
                                  std::vector<double> &gpuIntensity, std::vector<GPUFilter> &gpuFilter) :
        m_index(-1),
        m_type(""),
        m_algorithm(algorithm),
        m_variant(variant),
        m_priority(priority),
        m_cpuThreads(cpuThreads),
        m_cpuAffinity(cpuAffinity),
        m_cpuOptimization(cpuOptimization),
        m_gpuIntensity(gpuIntensity),
        m_gpuFilter(gpuFilter){

}

xmrig::HasherConfig::HasherConfig(int index, std::string type, xmrig::Algo algorithm, xmrig::Variant variant, int priority, int cpuThreads,
                                  int64_t cpuAffinity,  std::string cpuOptimization,
                                  std::vector<double> &gpuIntensity, std::vector<GPUFilter> &gpuFilter) :
        m_index(index),
        m_type(type),
        m_algorithm(algorithm),
        m_variant(variant),
        m_priority(priority),
        m_cpuThreads(cpuThreads),
        m_cpuAffinity(cpuAffinity),
        m_cpuOptimization(cpuOptimization),
        m_gpuIntensity(gpuIntensity) {
    for(GPUFilter filter : gpuFilter) {
        if(filter.engine.empty() || filter.engine == "*" || filter.engine == type) {
            m_gpuFilter.push_back(filter);
        }
    }
}

double xmrig::HasherConfig::getGPUIntensity(int cardIndex) {
    if(cardIndex < m_gpuIntensity.size())
        return m_gpuIntensity[cardIndex];
    else if(m_gpuIntensity.size() > 0)
        return m_gpuIntensity[0];
    else
        return 50;
}

int64_t xmrig::HasherConfig::getCPUAffinity(int cpuIndex) {
    int64_t cpuId = -1L;

    if (m_cpuAffinity != -1L) {
        size_t idx = 0;

        for (size_t i = 0; i < 64; i++) {
            if (!(m_cpuAffinity & (1ULL << i))) {
                continue;
            }

            if (idx == cpuIndex) {
                cpuId = i;
                break;
            }

            idx++;
        }
    }

    return cpuId;
}

xmrig::HasherConfig *xmrig::HasherConfig::clone(int index, std::string hasherType) {
    return new HasherConfig(index, hasherType, m_algorithm, m_variant, m_priority, m_cpuThreads, m_cpuAffinity, m_cpuOptimization, m_gpuIntensity, m_gpuFilter);
}

double xmrig::HasherConfig::getAverageGPUIntensity() {
    double result = 0;
    for(double intensity : m_gpuIntensity) result += intensity;
    return result / (m_gpuIntensity.size() > 0 ? m_gpuIntensity.size() : 1);
}

