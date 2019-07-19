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

#ifndef XMRIG_THREAD_H
#define XMRIG_THREAD_H


#include <uv.h>


#include "backend/common/interfaces/IWorker.h"


namespace xmrig {


class IBackend;


template<class T>
class Thread
{
public:
    inline Thread(IBackend *backend, size_t index, const T &config) : m_index(index), m_config(config), m_backend(backend) {}
    inline ~Thread() { uv_thread_join(&m_thread); delete m_worker; }

    inline const T &config() const                  { return m_config; }
    inline IBackend *backend() const                { return m_backend; }
    inline IWorker *worker() const                  { return m_worker; }
    inline size_t index() const                     { return m_index; }
    inline void setWorker(IWorker *worker)          { m_worker = worker; }
    inline void start(void (*callback) (void *))    { uv_thread_create(&m_thread, callback, this); }

private:
    const size_t m_index    = 0;
    const T m_config;
    IBackend *m_backend;
    IWorker *m_worker       = nullptr;
    uv_thread_t m_thread;
};


} // namespace xmrig


#endif /* XMRIG_THREAD_H */
