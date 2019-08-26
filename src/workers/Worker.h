/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include <stdint.h>


#include "interfaces/IWorker.h"
#include "common/net/Job.h"
#include "net/JobResult.h"

class Handle;


namespace xmrig {
    class HasherConfig;
}


class Worker : public IWorker
{
public:
    Worker(Handle *handle, int workerIdx);

    inline size_t id() const override          { return m_id; }
    inline uint64_t hashCount() const override { return m_hashCount.load(std::memory_order_relaxed); }
    inline uint64_t timestamp() const override { return m_timestamp.load(std::memory_order_relaxed); }
    inline size_t parallelism() const override { return m_hasher->parallelism(m_id); }

    bool selfTest() override;
    void start() override;

private:
    void storeStats();
    bool consumeJob();

    bool resume(const xmrig::Job &job);
    void save(const xmrig::Job &job);

    struct State
    {
        alignas(16) uint8_t blob[xmrig::Job::kMaxBlobSize];
        xmrig::Job job;
    };

    const size_t m_id;
    uint32_t m_offset;
    std::atomic<uint64_t> m_hashCount;
    std::atomic<uint64_t> m_timestamp;
    Hasher *m_hasher;
    uint64_t m_count;
    uint64_t m_sequence;
    xmrig::HasherConfig *m_config;
    State m_pausedState;
    State m_state;
    uint8_t *m_hash;
};


#endif /* XMRIG_WORKER_H */
