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
#include "base/tools/Object.h"


#ifdef XMRIG_FEATURE_OPENCL
#   include "backend/opencl/OclWorker.h"
#endif


#ifdef XMRIG_FEATURE_CUDA
#   include "backend/cuda/CudaWorker.h"
#endif


namespace xmrig {


class WorkersPrivate
{
public:
    XMRIG_DISABLE_COPY_MOVE(WorkersPrivate)


    WorkersPrivate() = default;


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
    Nonce::touch(T::backend());

    for (Thread<T> *worker : m_workers) {
        worker->start(Workers<T>::onReady);

        // This sleep is important for optimal caching!
        // Threads must allocate scratchpads in order so that adjacent cores will use adjacent scratchpads
        // Sub-optimal caching can result in up to 0.5% hashrate penalty
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
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
            continue;
        }

        d_ptr->hashrate->add(handle->id(), handle->worker()->hashCount(), handle->worker()->timestamp());
    }
}


template<class T>
xmrig::IWorker *xmrig::Workers<T>::create(Thread<T> *)
{
    return nullptr;
}


template<class T>
void xmrig::Workers<T>::onReady(void *arg)
{
    auto handle = static_cast<Thread<T>* >(arg);

    IWorker *worker = create(handle);
    assert(worker != nullptr);

    if (!worker || !worker->selfTest()) {
        LOG_ERR("%s " RED("thread ") RED_BOLD("#%zu") RED(" self-test failed"), T::tag(), worker ? worker->id() : 0);

        handle->backend()->start(worker, false);
        delete worker;

        return;
    }

    assert(handle->backend() != nullptr);

    handle->setWorker(worker);
    handle->backend()->start(worker, true);
}


namespace xmrig {


template<>
xmrig::IWorker *xmrig::Workers<CpuLaunchData>::create(Thread<CpuLaunchData> *handle)
{
    switch (handle->config().intensity) {
    case 1:
        return new CpuWorker<1>(handle->id(), handle->config());

    case 2:
        return new CpuWorker<2>(handle->id(), handle->config());

    case 3:
        return new CpuWorker<3>(handle->id(), handle->config());

    case 4:
        return new CpuWorker<4>(handle->id(), handle->config());

    case 5:
        return new CpuWorker<5>(handle->id(), handle->config());
    }

    return nullptr;
}


template class Workers<CpuLaunchData>;


#ifdef XMRIG_FEATURE_OPENCL
template<>
xmrig::IWorker *xmrig::Workers<OclLaunchData>::create(Thread<OclLaunchData> *handle)
{
    return new OclWorker(handle->id(), handle->config());
}


template class Workers<OclLaunchData>;
#endif


#ifdef XMRIG_FEATURE_CUDA
template<>
xmrig::IWorker *xmrig::Workers<CudaLaunchData>::create(Thread<CudaLaunchData> *handle)
{
    return new CudaWorker(handle->id(), handle->config());
}


template class Workers<CudaLaunchData>;
#endif


} // namespace xmrig
