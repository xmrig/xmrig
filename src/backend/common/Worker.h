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

#ifndef XMRIG_WORKER_H
#define XMRIG_WORKER_H


#include <atomic>
#include <cstdint>


#include "backend/common/interfaces/IWorker.h"


namespace xmrig {


class Worker : public IWorker
{
public:
    Worker(size_t id, int64_t affinity, int priority);

    inline const VirtualMemory *memory() const override   { return nullptr; }
    inline size_t id() const override                     { return m_id; }
    inline uint64_t rawHashes() const override            { return m_count; }
    inline void jobEarlyNotification(const Job&) override {}

    void getHashrateData(uint64_t& hashCount, uint64_t& timeStamp) const override;

#   ifdef XMRIG_FEATURE_BENCHMARK
    inline uint64_t benchData() const override            { return m_benchData; }
    inline uint64_t benchDoneTime() const override        { return m_benchDoneTime; }
#   endif

protected:
    void storeStats();

    const int64_t m_affinity;
    const size_t m_id;
    std::atomic<uint32_t> m_index   = {};
    uint32_t m_node                 = 0;
    uint64_t m_count                = 0;
    uint64_t m_hashCount[2]         = {};
    uint64_t m_timestamp[2]         = {};

#   ifdef XMRIG_FEATURE_BENCHMARK
    uint64_t m_benchData            = 0;
    uint64_t m_benchDoneTime        = 0;
#   endif
};


} // namespace xmrig


#endif /* XMRIG_WORKER_H */
