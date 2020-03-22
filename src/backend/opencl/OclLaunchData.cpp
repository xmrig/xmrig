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


#include "backend/opencl/OclLaunchData.h"
#include "backend/common/Tags.h"
#include "backend/opencl/OclConfig.h"


xmrig::OclLaunchData::OclLaunchData(const Miner *miner, const Algorithm &algorithm, const OclConfig &config, const OclPlatform &platform, const OclThread &thread, const OclDevice &device, int64_t affinity) :
    algorithm(algorithm),
    cache(config.isCacheEnabled()),
    affinity(affinity),
    miner(miner),
    device(device),
    platform(platform),
    thread(thread)
{
}


bool xmrig::OclLaunchData::isEqual(const OclLaunchData &other) const
{
    return (other.algorithm == algorithm &&
            other.thread    == thread);
}


const char *xmrig::OclLaunchData::tag()
{
    return ocl_tag();
}
