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


#include "backend/common/Hashrate.h"
#include "backend/common/interfaces/IBackend.h"
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
        delete hashrate;
    }


    Hashrate *hashrate = nullptr;
    IBackend *backend  = nullptr;
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
const xmrig::Hashrate *xmrig::Workers<T>::hashrate() const
{
    return d_ptr->hashrate;
}


template<class T>
void xmrig::Workers<T>::setBackend(IBackend *backend)
{
    d_ptr->backend = backend;
}


template<class T>
void xmrig::Workers<T>::start(const std::vector<T> &data)
{
    for (const T &item : data) {
        m_workers.push_back(new Thread<T>(d_ptr->backend, m_workers.size(), item));
    }

    d_ptr->hashrate = new Hashrate(m_workers.size());

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

    delete d_ptr->hashrate;
    d_ptr->hashrate = nullptr;
}


template<class T>
void xmrig::Workers<T>::tick(uint64_t)
{
    if (!d_ptr->hashrate) {
        return;
    }

    for (Thread<T> *handle : m_workers) {
        if (!handle->worker()) {
            return;
        }

        d_ptr->hashrate->add(handle->index(), handle->worker()->hashCount(), handle->worker()->timestamp());
    }

    d_ptr->hashrate->updateHighest();
}


template<class T>
xmrig::IWorker *xmrig::Workers<T>::create(Thread<CpuLaunchData> *)
{
    return nullptr;
}


template<class T>
void xmrig::Workers<T>::onReady(void *arg)
{
    Thread<T> *handle = static_cast<Thread<T>* >(arg);

    IWorker *worker = create(handle);
    if (!worker || !worker->selfTest()) {
        LOG_ERR("thread %zu error: \"hash self-test failed\".", worker->id());

        return;
    }

    handle->setWorker(worker);
    handle->backend()->start(worker);
}


namespace xmrig {


#if defined (XMRIG_ALGO_RANDOMX) || defined (XMRIG_ALGO_CN_GPU)
static void printIntensityWarning(Thread<CpuLaunchData> *handle)
{
    LOG_WARN("CPU thread %zu warning: \"intensity %d not supported for %s algorithm\".", handle->index(), handle->config().intensity, handle->config().algorithm.shortName());
}
#endif


template<>
xmrig::IWorker *xmrig::Workers<CpuLaunchData>::create(Thread<CpuLaunchData> *handle)
{
    const int intensity = handle->config().intensity;

#   if defined (XMRIG_ALGO_RANDOMX) || defined (XMRIG_ALGO_CN_GPU)
    if (intensity > 1) {
#       ifdef XMRIG_ALGO_RANDOMX
        if (handle->config().algorithm.family() == Algorithm::RANDOM_X) {
            printIntensityWarning(handle);

            return new CpuWorker<1>(handle->index(), handle->config());
        }
#       endif

#       ifdef XMRIG_ALGO_CN_GPU
        if (handle->config().algorithm == Algorithm::CN_GPU) {
            printIntensityWarning(handle);

            return new CpuWorker<1>(handle->index(), handle->config());
        }
#       endif
    }
#   endif


    switch (intensity) {
    case 1:
        return new CpuWorker<1>(handle->index(), handle->config());

    case 2:
        return new CpuWorker<2>(handle->index(), handle->config());

    case 3:
        return new CpuWorker<3>(handle->index(), handle->config());

    case 4:
        return new CpuWorker<4>(handle->index(), handle->config());

    case 5:
        return new CpuWorker<5>(handle->index(), handle->config());
    }

    return nullptr;
}


template class Workers<CpuLaunchData>;


} // namespace xmrig
