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
#include "crypto/randomx/randomx.h"
#include "crypto/rx/RxAlgo.h"


namespace xmrig {


bool ocl_generic_rx_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads)
{
    if (algorithm.family() != Algorithm::RANDOM_X) {
        return false;
    }

    // Mobile Ryzen APUs
    if (device.type() == OclDevice::Raven) {
        threads.add(OclThread(device.index(), (device.computeUnits() > 4) ? 256 : 128, 8, 1, true, true, 6));
        return true;
    }

    const size_t mem = device.globalMemSize();
    auto config      = RxAlgo::base(algorithm);
    bool gcnAsm      = false;
    bool isNavi      = false;

    switch (device.type()) {
    case OclDevice::Baffin:
    case OclDevice::Polaris:
    case OclDevice::Lexa:
    case OclDevice::Vega_10:
    case OclDevice::Vega_20:
        gcnAsm = true;
        break;

    case OclDevice::Navi_10:
    case OclDevice::Navi_12:
    case OclDevice::Navi_14:
        gcnAsm = true;
        isNavi = true;
        break;

    default:
        break;
    }

    // Must have space for dataset, scratchpads and 128 MB of free memory
    const uint32_t dataset_mem = config->DatasetBaseSize + config->DatasetExtraSize + (128U << 20);

    // Use dataset on host if not enough memory
    bool datasetHost = (mem < dataset_mem);

    // Each thread uses 1 scratchpad plus a few small buffers on GPU
    const uint32_t per_thread_mem = config->ScratchpadL3_Size + 32768;

    uint32_t intensity = static_cast<uint32_t>((mem - (datasetHost ? 0 : dataset_mem)) / per_thread_mem / 2);

    // Too high intensity makes hashrate worse
    const uint32_t intensityCoeff = isNavi ? 64 : 16;
    if (intensity > device.computeUnits() * intensityCoeff) {
        intensity = device.computeUnits() * intensityCoeff;
    }

    intensity -= intensity % 64;

    // If there are too few threads, use dataset on host to get more threads
    if (intensity < device.computeUnits() * 4) {
        datasetHost = true;
        intensity = static_cast<uint32_t>(mem / per_thread_mem / 2);
        intensity -= intensity % 64;
    }

    if (!intensity) {
        return false;
    }

    threads.add(OclThread(device.index(), intensity, 8, device.vendorId() == OCL_VENDOR_AMD ? 2 : 1, gcnAsm, datasetHost, 6));

    return true;
}


} // namespace xmrig
