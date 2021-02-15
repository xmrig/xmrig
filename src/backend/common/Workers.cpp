/* xmlcore
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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
#include "backend/common/Hashrate.h"
#include "backend/common/interfaces/IBackend.h"
#include "backend/cpu/CpuWorker.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/tools/Chrono.h"


#ifdef xmlcore_FEATURE_OPENCL
#   include "backend/opencl/OclWorker.h"
#endif


#ifdef xmlcore_FEATURE_CUDA
#   include "backend/cuda/CudaWorker.h"
#endif


#ifdef xmlcore_FEATURE_BENCHMARK
#   include "backend/common/benchmark/Benchmark.h"
#endif


namespace xmlcore {


class WorkersPrivate
{
public:
    xmlcore_DISABLE_COPY_MOVE(WorkersPrivate)

    WorkersPrivate()    = default;
    ~WorkersPrivate()   = default;

    IBackend *backend   = nullptr;
    std::shared_ptr<Benchmark> benchmark;
    std::shared_ptr<Hashrate> hashrate;
};


} // namespace xmlcore


template<class T>
xmlcore::Workers<T>::Workers() :
    d_ptr(new WorkersPrivate())
{

}


template<class T>
xmlcore::Workers<T>::~Workers()
{
    delete d_ptr;
}


template<class T>
bool xmlcore::Workers<T>::tick(uint64_t)
{
    if (!d_ptr->hashrate) {
        return true;
    }

    uint64_t ts             = Chrono::steadyMSecs();
    bool totalAvailable     = true;
    uint64_t totalHashCount = 0;
    uint64_t hashCount      = 0;
    uint64_t rawHashes      = 0;

    for (Thread<T> *handle : m_workers) {
        IWorker *worker = handle->worker();
        if (worker) {
            worker->hashrateData(hashCount, ts, rawHashes);
            d_ptr->hashrate->add(handle->id(), hashCount, ts);

            if (rawHashes == 0) {
                totalAvailable = false;
            }

            totalHashCount += rawHashes;
        }
    }

    if (totalAvailable) {
        d_ptr->hashrate->add(totalHashCount, Chrono::steadyMSecs());
    }

#   ifdef xmlcore_FEATURE_BENCHMARK
    return !d_ptr->benchmark || !d_ptr->benchmark->finish(totalHashCount);
#   else
    return true;
#   endif
}


template<class T>
const xmlcore::Hashrate *xmlcore::Workers<T>::hashrate() const
{
    return d_ptr->hashrate.get();
}


template<class T>
void xmlcore::Workers<T>::setBackend(IBackend *backend)
{
    d_ptr->backend = backend;
}


template<class T>
void xmlcore::Workers<T>::stop()
{
#   ifdef xmlcore_MINER_PROJECT
    Nonce::stop(T::backend());
#   endif

    for (Thread<T> *worker : m_workers) {
        delete worker;
    }

    m_workers.clear();

#   ifdef xmlcore_MINER_PROJECT
    Nonce::touch(T::backend());
#   endif

    d_ptr->hashrate.reset();
}


#ifdef xmlcore_FEATURE_BENCHMARK
template<class T>
void xmlcore::Workers<T>::start(const std::vector<T> &data, const std::shared_ptr<Benchmark> &benchmark)
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
xmlcore::IWorker *xmlcore::Workers<T>::create(Thread<T> *)
{
    return nullptr;
}


template<class T>
void *xmlcore::Workers<T>::onReady(void *arg)
{
    auto handle = static_cast<Thread<T>* >(arg);

    IWorker *worker = create(handle);
    assert(worker != nullptr);

    if (!worker || !worker->selfTest()) {
        LOG_ERR("%s " RED("thread ") RED_BOLD("#%zu") RED(" self-test failed"), T::tag(), worker ? worker->id() : 0);

        handle->backend()->start(worker, false);
        delete worker;

        return nullptr;
    }

    assert(handle->backend() != nullptr);

    handle->setWorker(worker);
    handle->backend()->start(worker, true);

    return nullptr;
}


template<class T>
void xmlcore::Workers<T>::start(const std::vector<T> &data, bool sleep)
{
    for (const auto &item : data) {
        m_workers.push_back(new Thread<T>(d_ptr->backend, m_workers.size(), item));
    }

    d_ptr->hashrate = std::make_shared<Hashrate>(m_workers.size());

#   ifdef xmlcore_MINER_PROJECT
    Nonce::touch(T::backend());
#   endif

    for (auto worker : m_workers) {
        worker->start(Workers<T>::onReady);
    }
}


namespace xmlcore {


template<>
xmlcore::IWorker *xmlcore::Workers<CpuLaunchData>::create(Thread<CpuLaunchData> *handle)
{
#   ifdef xmlcore_MINER_PROJECT
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
#   else
    assert(handle->config().intensity == 1);

    return new CpuWorker<1>(handle->id(), handle->config());
#   endif
}


template class Workers<CpuLaunchData>;


#ifdef xmlcore_FEATURE_OPENCL
template<>
xmlcore::IWorker *xmlcore::Workers<OclLaunchData>::create(Thread<OclLaunchData> *handle)
{
    return new OclWorker(handle->id(), handle->config());
}


template class Workers<OclLaunchData>;
#endif


#ifdef xmlcore_FEATURE_CUDA
template<>
xmlcore::IWorker *xmlcore::Workers<CudaLaunchData>::create(Thread<CudaLaunchData> *handle)
{
    return new CudaWorker(handle->id(), handle->config());
}


template class Workers<CudaLaunchData>;
#endif


} // namespace xmlcore
