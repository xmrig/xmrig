#include <thread>
#include "crypto/CryptoNight.h"
#include "workers/SingleWorker.h"
#include "workers/Workers.h"
SingleWorker::SingleWorker(Handle *handle)
    : Worker(handle)
{
}
void SingleWorker::start()
{
    while (Workers::sequence() > 0) {
        if (Workers::isPaused()) {
            do { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }
            while (Workers::isPaused());
            if (Workers::sequence() == 0) { break; }
            consumeJob();
        }
        while (!Workers::isOutdated(m_sequence)) {
            if ((m_count & 0xF) == 0) { storeStats(); }
            m_count++;
            *m_job.nonce() = ++m_result.nonce;
            if (CryptoNight::hash(m_job, m_result, m_ctx)) { Workers::submit(m_result); }
            std::this_thread::yield();
        }
        consumeJob();
    }
}
bool SingleWorker::resume(const Job &job)
{
    if (m_job.poolId() == -1 && job.poolId() >= 0 && job.id() == m_paused.id()) {
        m_job          = m_paused;
        m_result       = m_job;
        m_result.nonce = *m_job.nonce();
        return true;
    }
    return false;
}
void SingleWorker::consumeJob()
{
    Job job = Workers::job();
    m_sequence = Workers::sequence();
    if (m_job == job) { return; }
    save(job);
    if (resume(job)) { return; }
    m_job = std::move(job);
    m_result = m_job;
    if (m_job.isNicehash()) { m_result.nonce = (*m_job.nonce() & 0xff000000U) + (0xffffffU / m_threads * m_id); }
    else { m_result.nonce = 0xffffffffU / m_threads * m_id; }
}
void SingleWorker::save(const Job &job){ if (job.poolId() == -1 && m_job.poolId() >= 0) { m_paused = m_job; }}