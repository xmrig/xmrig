/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_THREAD_H
#define XMRIG_THREAD_H


#include "backend/common/interfaces/IWorker.h"


#include <thread>


#ifdef XMRIG_OS_APPLE
#   include <pthread.h>
#   include <mach/thread_act.h>
#endif


namespace xmrig {


class IBackend;


template<class T>
class Thread
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Thread)

    inline Thread(IBackend *backend, size_t id, const T &config) : m_id(id), m_config(config), m_backend(backend) {}

#   ifdef XMRIG_OS_APPLE
    inline ~Thread() { pthread_join(m_thread, nullptr); delete m_worker; }

    inline void start(void *(*callback)(void *))
    {
        if (m_config.affinity >= 0) {
            pthread_create_suspended_np(&m_thread, nullptr, callback, this);

            mach_port_t mach_thread              = pthread_mach_thread_np(m_thread);
            thread_affinity_policy_data_t policy = { static_cast<integer_t>(m_config.affinity + 1) };

            thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, reinterpret_cast<thread_policy_t>(&policy), THREAD_AFFINITY_POLICY_COUNT);
            thread_resume(mach_thread);
        }
        else {
            pthread_create(&m_thread, nullptr, callback, this);
        }
    }
#   else
    inline ~Thread() { m_thread.join(); delete m_worker; }

    inline void start(void *(*callback)(void *))    { m_thread = std::thread(callback, this); }
#   endif

    inline const T &config() const                  { return m_config; }
    inline IBackend *backend() const                { return m_backend; }
    inline IWorker *worker() const                  { return m_worker; }
    inline size_t id() const                        { return m_id; }
    inline void setWorker(IWorker *worker)          { m_worker = worker; }

private:
    const size_t m_id    = 0;
    const T m_config;
    IBackend *m_backend;
    IWorker *m_worker       = nullptr;

    #ifdef XMRIG_OS_APPLE
    pthread_t m_thread{};
#   else
    std::thread m_thread;
#   endif
};


} // namespace xmrig


#endif /* XMRIG_THREAD_H */
