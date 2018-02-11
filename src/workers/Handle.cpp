#include "workers/Handle.h"
Handle::Handle(int threadId, int threads, int64_t affinity, int priority) :
    m_priority(priority),
    m_threadId(threadId),
    m_threads(threads),
    m_affinity(affinity),
    m_worker(nullptr)
{
}
void Handle::join(){ uv_thread_join(&m_thread); }
void Handle::start(void (*callback) (void *)){ uv_thread_create(&m_thread, callback, this); }