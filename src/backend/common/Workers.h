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

#ifndef XMRIG_WORKERS_H
#define XMRIG_WORKERS_H


#include "backend/common/Thread.h"
#include "backend/cpu/CpuLaunchData.h"


#ifdef XMRIG_FEATURE_OPENCL
#   include "backend/opencl/OclLaunchData.h"
#endif


#ifdef XMRIG_FEATURE_CUDA
#   include "backend/cuda/CudaLaunchData.h"
#endif


namespace xmrig {


class Hashrate;
class WorkersPrivate;
class Job;
class Benchmark;


template<class T>
class Workers
{
public:
    XMRIG_DISABLE_COPY_MOVE(Workers)

    Workers();
    ~Workers();

    Benchmark *benchmark() const;
    bool tick(uint64_t ticks);
    const Hashrate *hashrate() const;
    void jobEarlyNotification(const Job&);
    void setBackend(IBackend *backend);
    void start(const std::vector<T> &data);
    void stop();

private:
    static IWorker *create(Thread<T> *handle);
    static void onReady(void *arg);

    std::vector<Thread<T> *> m_workers;
    WorkersPrivate *d_ptr;
};


template<class T>
void xmrig::Workers<T>::jobEarlyNotification(const Job& job)
{
    for (Thread<T>* t : m_workers) {
        if (t->worker()) {
            t->worker()->jobEarlyNotification(job);
        }
    }
}


template<>
IWorker *Workers<CpuLaunchData>::create(Thread<CpuLaunchData> *handle);
extern template class Workers<CpuLaunchData>;


#ifdef XMRIG_FEATURE_OPENCL
template<>
IWorker *Workers<OclLaunchData>::create(Thread<OclLaunchData> *handle);
extern template class Workers<OclLaunchData>;
#endif


#ifdef XMRIG_FEATURE_CUDA
template<>
IWorker *Workers<CudaLaunchData>::create(Thread<CudaLaunchData> *handle);
extern template class Workers<CudaLaunchData>;
#endif


} // namespace xmrig


#endif /* XMRIG_WORKERS_H */
