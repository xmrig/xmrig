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

#ifndef XMRIG_JOBRESULT_H
#define XMRIG_JOBRESULT_H


#include <memory.h>
#include <cstdint>


#include "base/tools/String.h"
#include "base/net/stratum/Job.h"


namespace xmrig {


class JobResult
{
public:
    JobResult() = delete;

    inline JobResult(const Job &job, uint64_t nonce, const uint8_t *result, const uint8_t* header_hash = nullptr, const uint8_t *mix_hash = nullptr, const uint8_t* miner_signature = nullptr) :
        algorithm(job.algorithm()),
        clientId(job.clientId()),
        jobId(job.id()),
        backend(job.backend()),
        nonce(nonce),
        diff(job.diff()),
        index(job.index())
    {
        memcpy(m_result, result, sizeof(m_result));

        if (header_hash) {
            memcpy(m_headerHash, header_hash, sizeof(m_headerHash));
        }

        if (mix_hash) {
            memcpy(m_mixHash, mix_hash, sizeof(m_mixHash));
        }

        if (miner_signature) {
            m_hasMinerSignature = true;
            memcpy(m_minerSignature, miner_signature, sizeof(m_minerSignature));
        }
    }

    inline JobResult(const Job &job) :
        algorithm(job.algorithm()),
        clientId(job.clientId()),
        jobId(job.id()),
        backend(job.backend()),
        nonce(0),
        diff(0),
        index(job.index())
    {
    }

    inline const uint8_t *result() const     { return m_result; }
    inline uint64_t actualDiff() const       { return Job::toDiff(reinterpret_cast<const uint64_t*>(m_result)[3]); }
    inline uint8_t *result()                 { return m_result; }
    inline const uint8_t *headerHash() const { return m_headerHash; }
    inline const uint8_t *mixHash() const    { return m_mixHash; }

    inline const uint8_t *minerSignature() const { return m_hasMinerSignature ? m_minerSignature : nullptr; }

    const Algorithm algorithm;
    const String clientId;
    const String jobId;
    const uint32_t backend;
    const uint64_t nonce;
    const uint64_t diff;
    const uint8_t index;

private:
    uint8_t m_result[32]     = { 0 };
    uint8_t m_headerHash[32] = { 0 };
    uint8_t m_mixHash[32]    = { 0 };

    uint8_t m_minerSignature[64] = { 0 };
    bool m_hasMinerSignature = false;
};


} /* namespace xmrig */


#endif /* XMRIG_JOBRESULT_H */
