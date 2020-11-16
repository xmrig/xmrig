/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include "base/io/log/Tags.h"
#include "base/net/stratum/Pool.h"
#include "base/tools/Chrono.h"
#include "base/tools/Object.h"
#include "core/Miner.h"


#ifdef XMRIG_FEATURE_OPENCL
#   include "backend/opencl/OclWorker.h"
#endif


#ifdef XMRIG_FEATURE_CUDA
#   include "backend/cuda/CudaWorker.h"
#endif


#ifdef XMRIG_FEATURE_BENCHMARK
#   include "backend/common/benchmark/Benchmark.h"
#endif


namespace xmrig {


class WorkersPrivate
{
public:
    XMRIG_DISABLE_COPY_MOVE(WorkersPrivate)


    WorkersPrivate()    = default;
    ~WorkersPrivate()   = default;

    IBackend *backend   = nullptr;
    std::shared_ptr<Benchmark> benchmark;
    std::shared_ptr<Hashrate> hashrate;
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
static void getHashrateData(xmrig::IWorker* worker, uint64_t& hashCount, uint64_t& timeStamp)
{
    worker->getHashrateData(hashCount, timeStamp);
}


template<>
void getHashrateData<xmrig::CpuLaunchData>(xmrig::IWorker* worker, uint64_t& hashCount, uint64_t&)
{
    hashCount = worker->rawHashes();
}


template<class T>
bool xmrig::Workers<T>::tick(uint64_t)
{
    if (!d_ptr->hashrate) {
        return true;
    }

    uint64_t ts             = Chrono::steadyMSecs();
    bool totalAvailable     = true;
    uint64_t totalHashCount = 0;

    for (Thread<T> *handle : m_workers) {
        IWorker *worker = handle->worker();
        if (worker) {
            uint64_t hashCount;
            getHashrateData<T>(worker, hashCount, ts);
            d_ptr->hashrate->add(handle->id() + 1, hashCount, ts);

            const uint64_t n = worker->rawHashes();
            if (n == 0) {
                totalAvailable = false;
            }
            totalHashCount += n;
        }
    }

    if (totalAvailable) {
        d_ptr->hashrate->add(0, totalHashCount, Chrono::steadyMSecs());
    }

#   ifdef XMRIG_FEATURE_BENCHMARK
    if (d_ptr->benchmark && d_ptr->benchmark->finish(totalHashCount)) {
        return false;
    }
#   endif

    return true;
}


template<class T>
const xmrig::Hashrate *xmrig::Workers<T>::hashrate() const
{
    return d_ptr->hashrate.get();
}


template<class T>
void xmrig::Workers<T>::setBackend(IBackend *backend)
{
    d_ptr->backend = backend;
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

    d_ptr->hashrate.reset();
}


#ifdef XMRIG_FEATURE_BENCHMARK
template<class T>
void xmrig::Workers<T>::start(const std::vector<T> &data, const std::shared_ptr<Benchmark> &benchmark)
{
    if (!benchmark) {
        return start(data, true);
    }

    start(data, false);

    d_ptr->benchmark = benchmark;
    d_ptr->benchmark->start();
}
#endif


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


template<class T>
void xmrig::Workers<T>::start(const std::vector<T> &data, bool sleep)
{
    for (const auto &item : data) {
        m_workers.push_back(new Thread<T>(d_ptr->backend, m_workers.size(), item));
    }

    d_ptr->hashrate = std::make_shared<Hashrate>(m_workers.size());
    Nonce::touch(T::backend());

    for (auto worker : m_workers) {
        worker->start(Workers<T>::onReady);

        // This sleep is important for optimal caching!
        // Threads must allocate scratchpads in order so that adjacent cores will use adjacent scratchpads
        // Sub-optimal caching can result in up to 0.5% hashrate penalty
        if (sleep) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
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
