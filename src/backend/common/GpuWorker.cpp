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


#include "backend/common/GpuWorker.h"
#include "base/tools/Chrono.h"


xmrig::GpuWorker::GpuWorker(size_t id, int64_t affinity, int priority, uint32_t deviceIndex) : Worker(id, affinity, priority),
    m_deviceIndex(deviceIndex)
{
}


void xmrig::GpuWorker::storeStats()
{
    // Get index which is unused now
    const uint32_t index = m_index.load(std::memory_order_relaxed) ^ 1;

    // Fill in the data for that index
    m_hashCount[index] = m_count;
    m_timestamp[index] = Chrono::steadyMSecs();

    // Switch to that index
    // All data will be in memory by the time it completes thanks to std::memory_order_seq_cst
    m_index.fetch_xor(1, std::memory_order_seq_cst);
}


void xmrig::GpuWorker::hashrateData(uint64_t &hashCount, uint64_t &timeStamp, uint64_t &rawHashes) const
{
    const uint32_t index = m_index.load(std::memory_order_relaxed);

    rawHashes = m_hashrateData.interpolate(timeStamp);
    hashCount = m_hashCount[index];
    timeStamp = m_timestamp[index];
}
