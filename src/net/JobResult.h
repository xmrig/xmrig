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

#ifndef XMRIG_JOBRESULT_H
#define XMRIG_JOBRESULT_H


#include <memory.h>
#include <stdint.h>


#include "base/tools/String.h"
#include "base/net/stratum/Job.h"


namespace xmrig {


class JobResult
{
public:
    inline JobResult() {}

    inline JobResult(const Job &job, uint32_t nonce, const uint8_t *result) :
        algorithm(job.algorithm()),
        clientId(job.clientId()),
        jobId(job.id()),
        nonce(nonce),
        diff(job.diff()),
        index(job.index())
    {
        memcpy(m_result, result, sizeof(m_result));
    }

    inline const uint8_t *result() const    { return m_result; }
    inline uint64_t actualDiff() const      { return Job::toDiff(reinterpret_cast<const uint64_t*>(m_result)[3]); }

    const Algorithm algorithm;
    const String clientId;
    const String jobId;
    const uint32_t nonce    = 0;
    const uint64_t diff     = 0;
    const uint8_t index     = 0;

private:
    uint8_t m_result[32];
};


} /* namespace xmrig */


#endif /* XMRIG_JOBRESULT_H */
