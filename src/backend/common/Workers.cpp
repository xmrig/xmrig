/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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


#include "backend/common/Workers.h"
#include "backend/cpu/CpuWorker.h"
#include "base/io/log/Log.h"


namespace xmrig {


class WorkersPrivate
{
public:
    inline WorkersPrivate()
    {

    }


    inline ~WorkersPrivate()
    {
    }
};


} // namespace xmrig


template<class T>
xmrig::Workers<T>::Workers() :
    d_ptr(new WorkersPrivate())
{

}


template<class T>
xmrig::Workers<T>::~Workers()
{
    delete d_ptr;
}


template<class T>
void xmrig::Workers<T>::add(const T &data)
{
    m_workers.push_back(new Thread<T>(m_workers.size(), data));
}


template<class T>
void xmrig::Workers<T>::start()
{
    for (Thread<T> *worker : m_workers) {
        worker->start(Workers<T>::onReady);
    }
}


template<class T>
void xmrig::Workers<T>::stop()
{
    Nonce::stop(T::backend());

    for (Thread<T> *worker : m_workers) {
        delete worker;
    }

    m_workers.clear();
    Nonce::touch(T::backend());
}


template<class T>
void xmrig::Workers<T>::onReady(void *arg)
{
    printf("ON READY\n");
}


namespace xmrig {


template<>
void xmrig::Workers<CpuLaunchData>::onReady(void *arg)
{
    auto handle = static_cast<Thread<CpuLaunchData>* >(arg);

    IWorker *worker = nullptr;

    switch (handle->config().intensity) {
    case 1:
        worker = new CpuWorker<1>(handle->index(), handle->config());
        break;

    case 2:
        worker = new CpuWorker<2>(handle->index(), handle->config());
        break;

    case 3:
        worker = new CpuWorker<3>(handle->index(), handle->config());
        break;

    case 4:
        worker = new CpuWorker<4>(handle->index(), handle->config());
        break;

    case 5:
        worker = new CpuWorker<5>(handle->index(), handle->config());
        break;
    }

    handle->setWorker(worker);

    if (!worker->selfTest()) {
        LOG_ERR("thread %zu error: \"hash self-test failed\".", handle->worker()->id());

        return;
    }

    worker->start();
}


template class Workers<CpuLaunchData>;


} // namespace xmrig
