/* XMRig
 * Copyright (c) 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright (c) 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include "crypto/cn/CnAlgo.h"


#include <algorithm>


namespace xmrig {


constexpr const size_t oneMiB = 1024U * 1024U;


static inline uint32_t getMaxThreads(const OclDevice &device, const Algorithm &algorithm)
{
    if (device.vendorId() == OCL_VENDOR_NVIDIA && (device.name().contains("P100") || device.name().contains("V100"))) {
        return 40000U;
    }

    const uint32_t ratio = (algorithm.l3() <= oneMiB) ? 2U : 1U;

    if (device.vendorId() == OCL_VENDOR_INTEL) {
        return ratio * device.computeUnits() * 8;
    }

    return ratio * 1000U;
}


static inline uint32_t getPossibleIntensity(const OclDevice &device, const Algorithm &algorithm)
{
    const uint32_t maxThreads   = getMaxThreads(device, algorithm);
    const size_t minFreeMem     = (maxThreads == 40000U ? 512U : 128U) * oneMiB;
    const size_t availableMem   = device.freeMemSize() - minFreeMem;
    const size_t perThread      = algorithm.l3() + 224U;
    const auto maxIntensity     = static_cast<uint32_t>(availableMem / perThread);

    return std::min<uint32_t>(maxThreads, maxIntensity);
}


static uint32_t getIntensity(const OclDevice &device, const Algorithm &algorithm)
{
    if (device.type() == OclDevice::Raven) {
        return 0;
    }

    const uint32_t maxIntensity = getPossibleIntensity(device, algorithm);

    uint32_t intensity = (maxIntensity / (8 * device.computeUnits())) * device.computeUnits() * 8;
    if (intensity == 0) {
        return 0;
    }

    if (device.vendorId() == OCL_VENDOR_AMD && (device.type() == OclDevice::Lexa || device.type() == OclDevice::Baffin || device.computeUnits() <= 16)) {
        intensity /= 2;

        if (algorithm.family() == Algorithm::CN_HEAVY) {
            intensity /= 2;
        }
    }

    return intensity;
}


static uint32_t getStridedIndex(const OclDevice &device, const Algorithm &algorithm)
{
    if (device.vendorId() != OCL_VENDOR_AMD) {
        return 0;
    }

    return algorithm.base() == Algorithm::CN_2 ? 2 : 1;
}


bool ocl_generic_cn_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads)
{
    if (!algorithm.isCN()) {
        return false;
    }

    const uint32_t intensity = getIntensity(device, algorithm);
    if (intensity == 0) {
        return false;
    }

    const uint32_t threadCount = (device.vendorId() == OCL_VENDOR_AMD && (device.globalMemSize() - intensity * 2 * algorithm.l3()) > 128 * oneMiB) ? 2 : 1;

    threads.add(OclThread(device.index(), intensity, 8, getStridedIndex(device, algorithm), 2, threadCount, 8));

    return true;
}


} // namespace xmrig
