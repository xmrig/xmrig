#ifndef __WORKERS_H__
#define __WORKERS_H__
#include <atomic>
#include <list>
#include <uv.h>
#include <vector>
#include "net/Job.h"
#include "net/JobResult.h"

class Handle;
class Hashrate;
class IJobResultListener;

class Workers
{
public:
    static Job job();
    static void printHashrate(bool detail);
    static void setEnabled(bool enabled);
    static void setJob(const Job &job);
    static void start(int64_t affinity, int priority);
    static void stop();
    static void submit(const JobResult &result);

    static inline bool isEnabled()                               { return m_enabled; }
    static inline bool isOutdated(uint64_t sequence)             { return m_sequence.load(std::memory_order_relaxed) != sequence; }
    static inline bool isPaused()                                { return m_paused.load(std::memory_order_relaxed) == 1; }
    static inline uint64_t sequence()                            { return m_sequence.load(std::memory_order_relaxed); }
    static inline void pause()                                   { m_active = false; m_paused = 1; m_sequence++; }
    static inline void setListener(IJobResultListener *listener) { m_listener = listener; }

private:
    static void onReady(void *arg);
    static void onResult(uv_async_t *handle);
    static void onTick(uv_timer_t *handle);

    static bool m_active;
    static bool m_enabled;
    static Hashrate *m_hashrate;
    static IJobResultListener *m_listener;
    static Job m_job;
    static std::atomic<int> m_paused;
    static std::atomic<uint64_t> m_sequence;
    static std::list<JobResult> m_queue;
    static std::vector<Handle*> m_workers;
    static uint64_t m_ticks;
    static uv_async_t m_async;
    static uv_mutex_t m_mutex;
    static uv_rwlock_t m_rwlock;
    static uv_timer_t m_timer;
};


#endif /* __WORKERS_H__ */
