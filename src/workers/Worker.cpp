#include <chrono>
#include "Cpu.h"
#include "Mem.h"
#include "Platform.h"
#include "workers/Handle.h"
#include "workers/Worker.h"
Worker::Worker(Handle *handle) :
    m_id(handle->threadId()),
    m_threads(handle->threads()),
    m_hashCount(0),
    m_timestamp(0),
    m_count(0),
    m_sequence(0)
{
    if (Cpu::threads() > 1 && handle->affinity() != -1L) {
        Cpu::setAffinity(m_id, handle->affinity());
    }

    Platform::setThreadPriority(handle->priority());
    m_ctx = Mem::create(m_id);
}
Worker::~Worker()
{
}
void Worker::storeStats()
{
    using namespace std::chrono;
    const uint64_t timestamp = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
    m_hashCount.store(m_count, std::memory_order_relaxed);
    m_timestamp.store(timestamp, std::memory_order_relaxed);
}
