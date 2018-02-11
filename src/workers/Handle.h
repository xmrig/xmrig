#ifndef __HANDLE_H__
#define __HANDLE_H__
#include <stdint.h>
#include <uv.h>

class IWorker;

class Handle
{
public:
    Handle(int threadId, int threads, int64_t affinity, int priority);
    void join();
    void start(void (*callback) (void *));

    inline int priority() const            { return m_priority; }
    inline int threadId() const            { return m_threadId; }
    inline int threads() const             { return m_threads; }
    inline int64_t affinity() const        { return m_affinity; }
    inline IWorker *worker() const         { return m_worker; }
    inline void setWorker(IWorker *worker) { m_worker = worker; }

private:
    int m_priority;
    int m_threadId;
    int m_threads;
    int64_t m_affinity;
    IWorker *m_worker;
    uv_thread_t m_thread;
};


#endif