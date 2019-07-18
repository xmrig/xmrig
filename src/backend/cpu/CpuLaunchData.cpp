/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include "backend/cpu/CpuLaunchData.h"
#include "backend/cpu/CpuConfig.h"


xmrig::CpuLaunchData::CpuLaunchData(const Miner *miner, const Algorithm &algorithm, const CpuConfig &config, const CpuThread &thread) :
    algorithm(algorithm),
    assembly(config.assembly()),
    hugePages(config.isHugePages()),
    hwAES(config.isHwAES()),
    intensity(thread.intensity()),
    priority(config.priority()),
    affinity(thread.affinity()),
    miner(miner)
{
}


bool xmrig::CpuLaunchData::isEqual(const CpuLaunchData &other) const
{
    return (algorithm.memory()  == other.algorithm.memory()
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
