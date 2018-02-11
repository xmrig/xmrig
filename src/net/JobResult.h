#ifndef __JOBRESULT_H__
#define __JOBRESULT_H__
#include <memory.h>
#include <stdint.h>
#include "Job.h"

class JobResult
{
public:
    inline JobResult() : poolId(0), diff(0), nonce(0) {}
    inline JobResult(int poolId, const JobId &jobId, uint32_t nonce, const uint8_t *result, uint32_t diff) :
        poolId(poolId),
        jobId(jobId),
        diff(diff),
        nonce(nonce)
    {
        memcpy(this->result, result, sizeof(this->result));
    }

    inline JobResult(const Job &job) : poolId(0), diff(0), nonce(0)
    {
        jobId  = job.id();
        poolId = job.poolId();
        diff   = job.diff();
        nonce  = *job.nonce();
    }

    inline JobResult &operator=(const Job &job) {
        jobId  = job.id();
        poolId = job.poolId();
        diff   = job.diff();

        return *this;
    }

    inline uint64_t actualDiff() const
    {
        return Job::toDiff(reinterpret_cast<const uint64_t*>(result)[3]);
    }

    int poolId;
    JobId jobId;
    uint32_t diff;
    uint32_t nonce;
    uint8_t result[32];
};

#endif