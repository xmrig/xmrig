#ifndef __DOUBLEWORKER_H__
#define __DOUBLEWORKER_H__
#include "align.h"
#include "net/Job.h"
#include "net/JobResult.h"
#include "workers/Worker.h"
class Handle;
class DoubleWorker : public Worker
{
public:
    DoubleWorker(Handle *handle);
    ~DoubleWorker();

    void start() override;

private:
    bool resume(const Job &job);
    void consumeJob();
    void save(const Job &job);

    class State;

    uint8_t m_hash[64];
    State *m_state;
    State *m_pausedState;
};

#endif