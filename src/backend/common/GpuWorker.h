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

#ifndef XMRIG_GPUWORKER_H
#define XMRIG_GPUWORKER_H


#include <atomic>


#include "backend/common/HashrateInterpolator.h"
#include "backend/common/Worker.h"


namespace xmrig {


class GpuWorker : public Worker
{
public:
    GpuWorker(size_t id, int64_t affinity, int priority, uint32_t m_deviceIndex);

protected:
    inline const VirtualMemory *memory() const override     { return nullptr; }
    inline uint32_t deviceIndex() const                     { return m_deviceIndex; }

    void hashrateData(uint64_t &hashCount, uint64_t &timeStamp, uint64_t &rawHashes) const override;

protected:
    void storeStats();

    const uint32_t m_deviceIndex;
    HashrateInterpolator m_hashrateData;
    std::atomic<uint32_t> m_index   = {};
    uint64_t m_hashCount[2]         = {};
    uint64_t m_timestamp[2]         = {};
};


} // namespace xmrig


#endif /* XMRIG_GPUWORKER_H */
