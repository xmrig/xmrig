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

#ifndef XMRIG_THREADHANDLE_H
#define XMRIG_THREADHANDLE_H


#include <assert.h>
#include <stdint.h>
#include <uv.h>


#include "interfaces/IThread.h"


class IWorker;


class ThreadHandle
{
public:
    ThreadHandle(xmrig::IThread *config, uint32_t offset, size_t totalWays);
    void join();
    void start(void (*callback) (void *));

    inline IWorker *worker() const         { return m_worker; }
    inline size_t threadId() const         { return m_config->index(); }
    inline size_t totalWays() const        { return m_totalWays; }
    inline uint32_t offset() const         { return m_offset; }
    inline void setWorker(IWorker *worker) { assert(worker != nullptr); m_worker = worker; }
    inline xmrig::IThread *config() const  { return m_config; }

private:
    IWorker *m_worker;
    size_t m_totalWays;
    uint32_t m_offset;
    uv_thread_t m_thread;
    xmrig::IThread *m_config;
};


#endif /* XMRIG_THREADHANDLE_H */
