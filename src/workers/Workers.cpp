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

#include "workers/Handle.h"
#include "workers/SingleWorker.h"
#include "workers/Workers.h"


Job Workers::m_job;
pthread_mutex_t Workers::m_mutex;
pthread_rwlock_t Workers::m_rwlock;
std::atomic<int> Workers::m_paused;
std::atomic<uint64_t> Workers::m_sequence;
std::list<JobResult> Workers::m_queue;
std::vector<Handle*> Workers::m_workers;
uv_async_t Workers::m_async;


Job Workers::job()
{
    pthread_rwlock_rdlock(&m_rwlock);
    Job job = m_job;
    pthread_rwlock_unlock(&m_rwlock);

    return std::move(job);
}


void Workers::setJob(const Job &job)
{
    pthread_rwlock_wrlock(&m_rwlock);
    m_job = job;
    pthread_rwlock_unlock(&m_rwlock);

    m_sequence++;
    m_paused = 0;
}


void Workers::start(int threads, int64_t affinity, bool nicehash)
{
    pthread_mutex_init(&m_mutex, nullptr);
    pthread_rwlock_init(&m_rwlock, nullptr);

    m_sequence = 0;
    m_paused   = 1;

    uv_async_init(uv_default_loop(), &m_async, Workers::onResult);

    for (int i = 0; i < threads; ++i) {
        Handle *handle = new Handle(i, threads, affinity, nicehash);
        m_workers.push_back(handle);
        handle->start(Workers::onReady);
    }
}


void Workers::submit(const JobResult &result)
{
    pthread_mutex_lock(&m_mutex);
    m_queue.push_back(result);
    pthread_mutex_unlock(&m_mutex);

    uv_async_send(&m_async);
}



void *Workers::onReady(void *arg)
{
    auto handle = static_cast<Handle*>(arg);
    IWorker *worker = new SingleWorker(handle);
    worker->start();

    return nullptr;
}



void Workers::onResult(uv_async_t *handle)
{
    std::list<JobResult> results;

    pthread_mutex_lock(&m_mutex);
    while (!m_queue.empty()) {
        results.push_back(std::move(m_queue.front()));
        m_queue.pop_front();
    }
    pthread_mutex_unlock(&m_mutex);
}
