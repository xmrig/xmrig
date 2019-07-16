/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_WORKERSLEGACY_H
#define XMRIG_WORKERSLEGACY_H


#include <atomic>
#include <list>
#include <uv.h>
#include <vector>

#ifdef XMRIG_ALGO_RANDOMX
#   include <randomx.h>
#endif

#include "backend/common/Thread.h"
#include "backend/cpu/CpuLaunchData.h"
#include "base/net/stratum/Job.h"
#include "net/JobResult.h"
#include "rapidjson/fwd.h"


//class Hashrate;


namespace xmrig {
    class IWorker;
    class Controller;
    class ThreadHandle;
}


class WorkersLegacy
{
public:
//    static size_t hugePages();
//    static size_t threads();
//    static void pause();
//    static void printHashrate(bool detail);
//    static void setEnabled(bool enabled);
//    static void setJob(const xmrig::Job &job, bool donate);
    static void start(xmrig::Controller *controller);
//    static void stop();
//    static xmrig::Job job();

//    static inline bool isEnabled()                                      { return m_enabled; }
//    static inline Hashrate *hashrate()                                  { return m_hashrate; }

//#   ifdef XMRIG_FEATURE_API
//    static void threadsSummary(rapidjson::Document &doc);
//#   endif

private:
//    static void onReady(void *arg);
//    static void onTick(uv_timer_t *handle);
    static void start(xmrig::IWorker *worker);

    class LaunchStatus
    {
    public:
        inline LaunchStatus() :
            hugePages(0),
            pages(0),
            started(0),
            threads(0),
            ways(0)
        {}

        size_t hugePages;
        size_t pages;
        size_t started;
        size_t threads;
        size_t ways;
        xmrig::Algorithm algo;
    };

    static bool m_active;
    static bool m_enabled;
//    static Hashrate *m_hashrate;
    static xmrig::Job m_job;
    static LaunchStatus m_status;
    static std::vector<xmrig::Thread<xmrig::CpuLaunchData>* > m_workers;
    static uint64_t m_ticks;
    static uv_mutex_t m_mutex;
    static uv_rwlock_t m_rwlock;
//    static uv_timer_t *m_timer;
    static xmrig::Controller *m_controller;
};


#endif /* XMRIG_WORKERSLEGACY_H */
