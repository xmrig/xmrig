/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CUDALAUNCHDATA_H
#define XMRIG_CUDALAUNCHDATA_H


#include "backend/cuda/CudaThread.h"
#include "base/crypto/Algorithm.h"
#include "crypto/common/Nonce.h"


namespace xmrig {


class CudaDevice;
class Miner;


class CudaLaunchData
{
public:
    CudaLaunchData(const Miner *miner, const Algorithm &algorithm, const CudaThread &thread, const CudaDevice &device);

    bool isEqual(const CudaLaunchData &other) const;

    inline constexpr static Nonce::Backend backend() { return Nonce::CUDA; }

    inline bool operator!=(const CudaLaunchData &other) const    { return !isEqual(other); }
    inline bool operator==(const CudaLaunchData &other) const    { return isEqual(other); }

    static const char *tag();

    const Algorithm algorithm;
    const CudaDevice &device;
    const CudaThread thread;
    const int64_t affinity;
    const Miner *miner;
    const uint32_t benchSize = 0;
};


} // namespace xmrig


#endif /* XMRIG_OCLLAUNCHDATA_H */
