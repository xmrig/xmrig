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


bool ocl_generic_astrobwt_generator(const OclDevice &device, const Algorithm &algorithm, OclThreads &threads)
{
    if (algorithm.family() != Algorithm::ASTROBWT) {
        return false;
    }

    if (algorithm.id() == Algorithm::ASTROBWT_DERO_2) {
        uint32_t intensity = device.computeUnits() * 128;
        if (!intensity || (intensity > 4096)) {
            intensity = 4096;
        }
        threads.add(OclThread(device.index(), intensity, 1));
        return true;
    }

    const size_t mem = device.globalMemSize();

    uint32_t per_thread_mem = 10 << 20;
    uint32_t intensity = static_cast<uint32_t>((mem - (128 << 20)) / per_thread_mem / 2);

    intensity &= ~63U;

    if (!intensity) {
        return false;
    }

    if (intensity > 256) {
        intensity = 256;
    }

    threads.add(OclThread(device.index(), intensity, 2));

    return true;
}


} // namespace xmrig
