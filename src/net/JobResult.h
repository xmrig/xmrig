/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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

#ifndef __JOBRESULT_H__
#define __JOBRESULT_H__


#include <memory.h>
#include <stdint.h>


#include "Job.h"


class JobResult
{
public:
    inline JobResult() : poolId(0), diff(0), nonce(0) {}
    inline JobResult(int poolId, const char *jobId, uint32_t nonce, const uint8_t *result, uint32_t diff) : poolId(poolId), diff(diff), nonce(nonce)
    {
        memcpy(this->jobId, jobId, sizeof(this->jobId));
        memcpy(this->result, result, sizeof(this->result));
    }


    inline JobResult(const Job &job) : poolId(0), diff(0), nonce(0)
    {
        memcpy(jobId, job.id(), sizeof(jobId));
        poolId = job.poolId();
        diff   = job.diff();
        nonce  = *job.nonce();
    }


    inline JobResult &operator=(const Job &job) {
        memcpy(jobId, job.id(), sizeof(jobId));
        poolId = job.poolId();
        diff   = job.diff();

        return *this;
    }


    char jobId[64];
    int poolId;
    uint32_t diff;
    uint32_t nonce;
    uint8_t result[32];
};

#endif /* __JOBRESULT_H__ */
