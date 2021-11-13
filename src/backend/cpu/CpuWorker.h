/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CPUWORKER_H
#define XMRIG_CPUWORKER_H


#include "backend/common/Worker.h"
#include "backend/common/WorkerJob.h"
#include "backend/cpu/CpuLaunchData.h"
#include "base/tools/Object.h"
#include "net/JobResult.h"


#ifdef XMRIG_ALGO_RANDOMX
class randomx_vm;
#endif


namespace xmrig {


class RxVm;


#ifdef XMRIG_ALGO_GHOSTRIDER
namespace ghostrider { struct HelperThread; }
#endif


template<size_t N>
class CpuWorker : public Worker
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(CpuWorker)

    CpuWorker(size_t id, const CpuLaunchData &data);
    ~CpuWorker() override;

protected:
    bool selfTest() override;
    void hashrateData(uint64_t &hashCount, uint64_t &timeStamp, uint64_t &rawHashes) const override;
    void start() override;

    inline const VirtualMemory *memory() const override     { return m_memory; }
    inline size_t intensity() const override                { return N; }
    inline void jobEarlyNotification(const Job&) override   {}

private:
    inline cn_hash_fun fn(const Algorithm &algorithm) const { return CnHash::fn(algorithm, m_av, m_assembly); }

#   ifdef XMRIG_ALGO_RANDOMX
    void allocateRandomX_VM();
#   endif

    bool nextRound();
    bool verify(const Algorithm &algorithm, const uint8_t *referenceValue);
    bool verify2(const Algorithm &algorithm, const uint8_t *referenceValue);
    void allocateCnCtx();
    void consumeJob();

    alignas(16) uint8_t m_hash[N * 32]{ 0 };
    const Algorithm m_algorithm;
    const Assembly m_assembly;
    const bool m_astrobwtAVX2;
    const bool m_hwAES;
    const bool m_yield;
    const CnHash::AlgoVariant m_av;
    const int m_astrobwtMaxSize;
    const Miner *m_miner;
    const size_t m_threads;
    cryptonight_ctx *m_ctx[N];
    VirtualMemory *m_memory = nullptr;
    WorkerJob<N> m_job;

#   ifdef XMRIG_ALGO_RANDOMX
    randomx_vm *m_vm        = nullptr;
#   endif

#   ifdef XMRIG_ALGO_GHOSTRIDER
    ghostrider::HelperThread* m_ghHelper = nullptr;
#   endif

#   ifdef XMRIG_FEATURE_BENCHMARK
    uint32_t m_benchSize    = 0;
#   endif
};


template<>
bool CpuWorker<1>::verify2(const Algorithm &algorithm, const uint8_t *referenceValue);


extern template class CpuWorker<1>;
extern template class CpuWorker<2>;
extern template class CpuWorker<3>;
extern template class CpuWorker<4>;
extern template class CpuWorker<5>;
extern template class CpuWorker<8>;


} // namespace xmrig


#endif /* XMRIG_CPUWORKER_H */
