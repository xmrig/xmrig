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


#include <assert.h>
#include <list>
#include <uv.h>


#include "base/tools/Handle.h"
#include "net/interfaces/IJobResultListener.h"
#include "net/JobResult.h"
#include "net/JobResults.h"


namespace xmrig {


class JobResultsPrivate
{
public:
    inline JobResultsPrivate()
    {
        uv_mutex_init(&m_mutex);

        m_async = new uv_async_t;
        m_async->data = this;

        uv_async_init(uv_default_loop(), m_async, JobResultsPrivate::onResult);
    }


    inline ~JobResultsPrivate()
    {
        Handle::close(m_async);

        uv_mutex_destroy(&m_mutex);
    }


    void setListener(IJobResultListener *listener)
    {
        m_listener = listener;
    }


    void submit(const JobResult &result)
    {
        uv_mutex_lock(&m_mutex);
        m_queue.push_back(result);
        uv_mutex_unlock(&m_mutex);

        uv_async_send(m_async);
    }


private:
    static void onResult(uv_async_t *handle)
    {
        static_cast<JobResultsPrivate*>(handle->data)->submit();
    }


    inline void submit()
    {
        std::list<JobResult> results;

        uv_mutex_lock(&m_mutex);
        while (!m_queue.empty()) {
            results.push_back(std::move(m_queue.front()));
            m_queue.pop_front();
        }
        uv_mutex_unlock(&m_mutex);

        for (auto result : results) {
            m_listener->onJobResult(result);
        }

        results.clear();
    }


    IJobResultListener *m_listener = nullptr;
    std::list<JobResult> m_queue;
    uv_async_t *m_async;
    uv_mutex_t m_mutex;
};


static JobResultsPrivate *handler = new JobResultsPrivate();


} // namespace xmrig



void xmrig::JobResults::setListener(IJobResultListener *listener)
{
    assert(handler != nullptr && listener != nullptr);

    handler->setListener(listener);
}


void xmrig::JobResults::stop()
{
    delete handler;

    handler = nullptr;
}


void xmrig::JobResults::submit(const JobResult &result)
{
    assert(handler != nullptr);

    if (handler) {
        handler->submit(result);
    }
}
