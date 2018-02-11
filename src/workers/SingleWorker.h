#ifndef __SINGLEWORKER_H__
#define __SINGLEWORKER_H__
#include "net/Job.h"
#include "net/JobResult.h"
#include "workers/Worker.h"

class Handle;

class SingleWorker : public Worker
{
public:
    SingleWorker(Handle *handle);

    void start() override;

private:
    bool resume(const Job &job);
    void consumeJob();
    void save(const Job &job);

    Job m_job;
    Job m_paused;
    JobResult m_result;
};


#endif