/* XMRig
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


static inline bool isMatch(const OclDevice &device, const Algorithm &algorithm)
{
    return algorithm.isCN() &&
           device.vendorId() == OCL_VENDOR_AMD &&
           (device.type() == OclDevice::Vega_10 || device.type() == OclDevice::Vega_20);
}


static inline uint32_t getMaxThreads(const OclDevice &device, const Algorithm &algorithm)
{
    const uint32_t ratio = (algorithm.l3() <= oneMiB) ? 2U : 1U;

    if (device.type() == OclDevice::Vega_10) {
        if (device.computeUnits() == 56 && algorithm.family() == Algorithm::CN && algorithm.base() == Algorithm::CN_2) {
            return 1792U;
        }
    }

    return ratio * 2024U;
}


static inline uint32_t getPossibleIntensity(const OclDevice &device, const Algorithm &algorithm)
{
    const uint32_t maxThreads   = getMaxThreads(device, algorithm);
    const size_t availableMem   = device.freeMemSize() - (128U * oneMiB);
    const size_t perThread      = algorithm.l3() + 224U;
    const auto maxIntensity     = static_cast<uint32_t>(availableMem / perThread);

    return std::min<uint32_t>(maxThreads, maxIntensity);
}


static inline uint32_t getIntensity(const OclDevice &device, const Algorithm &algorithm)
{
    const uint32_t maxIntensity = getPossibleIntensity(device, algorithm);

    if (device.type() == OclDevice::Vega_10) {
        if (algorithm.family() == Algorithm::CN_HEAVY && device.computeUnits() == 64 && maxIntensity > 976) {
            return 976;
        }
    }

    return maxIntensity / device.computeUnits() * device.computeUnits();
}


static inline uint32_t getWorksize(const Algorithm &algorithm)
{
    const auto f = algorithm.family();
    if (f == Algorithm::CN_PICO || f == Algorithm::CN_FEMTO) {
        return 64;
    }

    return algorithm.base() == Algorithm::CN_2 ? 16 : 8;
}


static uint32_t getStridedIndex(const Algorithm &algorithm)
{
    return algorithm.base() == Algorithm::CN_2 ? 2 : 1;
}


static inline uint32_t getMemChunk(const Algorithm &algorithm)
{
    return algorithm.base() == Algorithm::CN_2 ? 1 : 2;
}


bool ocl_vega_cn_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads)
{
    if (!isMatch(device, algorithm)) {
        return false;
    }

    const uint32_t intensity = getIntensity(device, algorithm);
    if (intensity == 0) {
        return false;
    }

    const uint32_t worksize = getWorksize(algorithm);
    const uint32_t memChunk = getMemChunk(algorithm);

    threads.add(OclThread(device.index(), intensity, worksize, getStridedIndex(algorithm), memChunk, 2, 8));

    return true;
}


} // namespace xmrig
