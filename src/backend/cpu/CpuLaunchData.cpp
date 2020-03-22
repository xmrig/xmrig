/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include "backend/cpu/CpuLaunchData.h"
#include "backend/common/Tags.h"
#include "backend/cpu/CpuConfig.h"


#include <algorithm>


xmrig::CpuLaunchData::CpuLaunchData(const Miner *miner, const Algorithm &algorithm, const CpuConfig &config, const CpuThread &thread) :
    algorithm(algorithm),
    assembly(config.assembly()),
    astrobwtAVX2(config.astrobwtAVX2()),
    hugePages(config.isHugePages()),
    hwAES(config.isHwAES()),
    yield(config.isYield()),
    astrobwtMaxSize(config.astrobwtMaxSize()),    
    priority(config.priority()),
    affinity(thread.affinity()),
    miner(miner),
    intensity(std::min<uint32_t>(thread.intensity(), algorithm.maxIntensity()))
{
}


bool xmrig::CpuLaunchData::isEqual(const CpuLaunchData &other) const
{
    return (algorithm.l3()      == other.algorithm.l3()
            && assembly         == other.assembly
            && hugePages        == other.hugePages
            && hwAES            == other.hwAES
            && intensity        == other.intensity
            && priority         == other.priority
            && affinity         == other.affinity
            );
}


xmrig::CnHash::AlgoVariant xmrig::CpuLaunchData::av() const
{
    if (intensity <= 2) {
        return static_cast<CnHash::AlgoVariant>(!hwAES ? (intensity + 2) : intensity);
    }

    return static_cast<CnHash::AlgoVariant>(!hwAES ? (intensity + 5) : (intensity + 2));
}


const char *xmrig::CpuLaunchData::tag()
{
    return cpu_tag();
}
