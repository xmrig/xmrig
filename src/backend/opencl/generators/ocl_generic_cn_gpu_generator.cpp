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


#include "backend/opencl/OclThreads.h"
#include "backend/opencl/wrappers/OclDevice.h"
#include "base/crypto/Algorithm.h"


#include <algorithm>


namespace xmrig {


constexpr const size_t oneMiB = 1024u * 1024u;



bool ocl_generic_cn_gpu_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads)
{
    if (algorithm != Algorithm::CN_GPU) {
        return false;
    }

    uint32_t worksize   = 8;
    uint32_t numThreads = 1u;
    size_t minFreeMem   = 128u * oneMiB;

    if (device.type() == OclDevice::Vega_10 || device.type() == OclDevice::Vega_20) {
        minFreeMem  = oneMiB;
        worksize    = 16;
    }
    else if (device.type() == OclDevice::Navi_10) {
        numThreads = 2u;
    }
    else if (device.name() == "Fiji") {
        worksize = 16;
    }

    size_t maxThreads = device.computeUnits() * 6 * 8;

    const size_t maxAvailableFreeMem = device.freeMemSize() - minFreeMem;
    const size_t memPerThread        = std::min(device.maxMemAllocSize(), maxAvailableFreeMem);

    size_t memPerHash           = algorithm.l3() + 240u;
    size_t maxIntensity         = memPerThread / memPerHash;
    size_t possibleIntensity    = std::min(maxThreads, maxIntensity);
    size_t intensity            = 0;
    size_t cuUtilization        = ((possibleIntensity * 100)  / (worksize * device.computeUnits())) % 100;

    if (cuUtilization >= 75) {
        intensity = (possibleIntensity / worksize) * worksize;
    }
    else {
        intensity = (possibleIntensity / (worksize * device.computeUnits())) * device.computeUnits() * worksize;
    }

    threads.add(OclThread(device.index(), intensity, worksize, numThreads, 1));

    return true;
}


} // namespace xmrig
