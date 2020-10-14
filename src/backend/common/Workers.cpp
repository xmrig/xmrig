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
#include "base/io/log/Tags.h"
#include "base/tools/Chrono.h"
#include "base/tools/Object.h"
#include "core/Miner.h"


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

    uint32_t bench      = 0;
    Algorithm benchAlgo = Algorithm::RX_0;
    uint64_t startTime  = 0;
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
    if (!data.empty()) {
        d_ptr->bench = data.front().miner->job().bench();
        d_ptr->benchAlgo = data.front().miner->job().algorithm();
    }

    for (const T &item : data) {
        m_workers.push_back(new Thread<T>(d_ptr->backend, m_workers.size(), item));
    }

    d_ptr->hashrate = new Hashrate(m_workers.size());
    Nonce::touch(T::backend());

    for (Thread<T> *worker : m_workers) {
        worker->start(Workers<T>::onReady);

        if (!d_ptr->bench) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    d_ptr->startTime = Chrono::steadyMSecs();
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
static void getHashrateData(xmrig::IWorker* worker, uint64_t& hashCount, uint64_t& timeStamp)
{
    worker->getHashrateData(hashCount, timeStamp);
}


template<>
static void getHashrateData<xmrig::CpuLaunchData>(xmrig::IWorker* worker, uint64_t& hashCount, uint64_t&)
{
    hashCount = worker->rawHashes();
}


template<class T>
bool xmrig::Workers<T>::tick(uint64_t)
{
    if (!d_ptr->hashrate) {
        return true;
    }

    uint64_t timeStamp = Chrono::steadyMSecs();

    bool totalAvailable = true;
    uint64_t totalHashCount = 0;

    uint32_t benchDone = 0;
    uint64_t benchData = 0;
    uint64_t benchDoneTime = 0;

    for (Thread<T> *handle : m_workers) {
        IWorker* worker = handle->worker();
        if (worker) {
            uint64_t hashCount;
            getHashrateData<T>(worker, hashCount, timeStamp);
            d_ptr->hashrate->add(handle->id() + 1, hashCount, timeStamp);

            const uint64_t n = worker->rawHashes();
            if (n == 0) {
                totalAvailable = false;
            }
            totalHashCount += n;

            if (d_ptr->bench && worker->benchDoneTime()) {
                ++benchDone;
                benchData ^= worker->benchData();
                if (worker->benchDoneTime() > benchDoneTime) {
                    benchDoneTime = worker->benchDoneTime();
                }
            }
        }
    }

    if (totalAvailable) {
        d_ptr->hashrate->add(0, totalHashCount, Chrono::steadyMSecs());
    }

    if (d_ptr->bench && (benchDone == m_workers.size())) {
        const double dt = (benchDoneTime - d_ptr->startTime) / 1000.0;

        static uint64_t hashCheck[Algorithm::MAX][2] = {};
        hashCheck[Algorithm::RX_0][0] = 0x898B6E0431C28A6BULL;
        hashCheck[Algorithm::RX_0][1] = 0xB5231262E2792B26ULL;
        hashCheck[Algorithm::RX_WOW][0] = 0x0F3E5400B39EA96AULL;
        hashCheck[Algorithm::RX_WOW][1] = 0x0F9E00C5A511C200ULL;

        int k = -1;

        switch (d_ptr->bench) {
        case 1000000:
            k = 0;
            break;

        case 10000000:
            k = 1;
            break;
        }

        const uint64_t checkData = (k >= 0) ? hashCheck[d_ptr->benchAlgo.id()][k] : 0;
        const char* color = checkData ? ((benchData == checkData) ? GREEN_BOLD_S : RED_BOLD_S) : BLACK_BOLD_S;

        LOG_INFO("%s Benchmark finished in %.3f seconds, hash sum = %s%016" PRIX64 CLEAR, Tags::miner(), dt, color, benchData);
        return false;
    }

    return true;
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
