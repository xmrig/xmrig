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

#ifndef XMRIG_MULTIWORKER_H
#define XMRIG_MULTIWORKER_H


#include <randomx.h>


#include "base/net/stratum/Job.h"
#include "Mem.h"
#include "net/JobResult.h"
#include "workers/Worker.h"


class Handle;


template<size_t N>
class MultiWorker : public Worker
{
public:
    MultiWorker(ThreadHandle *handle);
    ~MultiWorker();

protected:
    bool selfTest() override;
    void start() override;

private:
    void allocateRandomX_VM();

    bool resume(const xmrig::Job &job);
    bool verify(xmrig::Variant variant, const uint8_t *referenceValue);
    bool verify2(xmrig::Variant variant, const uint8_t *referenceValue);
    void consumeJob();
    void save(const xmrig::Job &job);

    inline uint32_t *nonce(size_t index)
    {
        return reinterpret_cast<uint32_t*>(m_state.blob + (index * m_state.job.size()) + 39);
    }

    struct State
    {
        alignas(16) uint8_t blob[xmrig::Job::kMaxBlobSize * N];
        xmrig::Job job;
    };


    cryptonight_ctx *m_ctx[N];
    State m_pausedState;
    State m_state;
    uint8_t m_hash[N * 32];

    randomx_vm* m_rx_vm;
};


#endif /* XMRIG_MULTIWORKER_H */
