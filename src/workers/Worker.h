#ifndef __WORKER_H__
#define __WORKER_H__
#include <atomic>
#include <stdint.h>
#include "interfaces/IWorker.h"
struct cryptonight_ctx;
class Handle;
class Worker : public IWorker
{
public:
    Worker(Handle *handle);
    ~Worker();

    inline uint64_t hashCount() const override { return m_hashCount.load(std::memory_order_relaxed); }
    inline uint64_t timestamp() const override { return m_timestamp.load(std::memory_order_relaxed); }

protected:
    void storeStats();

    cryptonight_ctx *m_ctx;
    int m_id;
    int m_threads;
    std::atomic<uint64_t> m_hashCount;
    std::atomic<uint64_t> m_timestamp;
    uint64_t m_count;
    uint64_t m_sequence;
};

#endif /* __WORKER_H__ */
