/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef __HANDLE_H__
#define __HANDLE_H__


#include <assert.h>
#include <stdint.h>
#include <vector>
#include <uv.h>
#include <core/Config.h>

#include "core/HasherConfig.h"

#include "crypto/argon2_hasher/common/common.h"
#include "crypto/argon2_hasher/hash/Hasher.h"

class IWorker;

class Handle
{
public:
    Handle(int id, xmrig::Config *config, xmrig::HasherConfig *hasherConfig, uint32_t offset);

    struct HandleArg {
        Handle *handle;
        int workerId;
    };

    void join();
    void start(void (*callback) (void *));

    inline std::vector<IWorker *> &workers()         { return m_workers; }
    inline size_t hasherId() const         { return m_id; }
    inline size_t parallelism(int workerIdx) const        { return m_hasher != nullptr ? m_hasher->parallelism(workerIdx) : 0; }
    inline size_t computingThreads() const   { return m_hasher != nullptr ? m_hasher->computingThreads() : 0; }
    inline uint32_t offset() const         { return m_offset; }
    inline void addWorker(IWorker *worker) { assert(worker != nullptr); m_workers.push_back(worker); }
    inline xmrig::HasherConfig *config() const  { return m_hasherConfig; }
    inline Hasher *hasher() const { return m_hasher; }

private:
    int m_id;
    std::vector<uv_thread_t> m_threads;
    std::vector<IWorker *> m_workers;

    Hasher *m_hasher;
    uint32_t m_offset;

    xmrig::HasherConfig *m_hasherConfig;
    xmrig::Config *m_config;
};


#endif /* __HANDLE_H__ */
