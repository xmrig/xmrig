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


#include <cassert>
#include <thread>


#include "backend/cpu/CpuWorker.h"
#include "core/Miner.h"
#include "crypto/cn/CnCtx.h"
#include "crypto/cn/CryptoNight_test.h"
#include "crypto/cn/CryptoNight.h"
#include "crypto/common/Nonce.h"
#include "crypto/common/VirtualMemory.h"
#include "crypto/rx/Rx.h"
#include "crypto/rx/RxVm.h"
#include "net/JobResults.h"


#ifdef XMRIG_ALGO_RANDOMX
#   include "crypto/randomx/randomx.h"
#endif


#ifdef XMRIG_ALGO_ASTROBWT
#   include "crypto/astrobwt/AstroBWT.h"
#endif


namespace xmrig {

static constexpr uint32_t kReserveCount = 32768;


template<size_t N>
inline bool nextRound(WorkerJob<N> &job)
{
    if (!job.nextRound(kReserveCount, 1)) {
        JobResults::done(job.currentJob());

        return false;
    }

    return true;
}


} // namespace xmrig



template<size_t N>
xmrig::CpuWorker<N>::CpuWorker(size_t id, const CpuLaunchData &data) :
    Worker(id, data.affinity, data.priority),
    m_algorithm(data.algorithm),
    m_assembly(data.assembly),
    m_astrobwtAVX2(data.astrobwtAVX2),
    m_hwAES(data.hwAES),
    m_yield(data.yield),
    m_av(data.av()),
    m_astrobwtMaxSize(data.astrobwtMaxSize * 1000),
    m_miner(data.miner),
    m_ctx()
{
    m_memory = new VirtualMemory(m_algorithm.l3() * N, data.hugePages, false, true, m_node);
}


template<size_t N>
xmrig::CpuWorker<N>::~CpuWorker()
{
#   ifdef XMRIG_ALGO_RANDOMX
    RxVm::destroy(m_vm);
#   endif

    CnCtx::release(m_ctx, N);
    delete m_memory;
}


#ifdef XMRIG_ALGO_RANDOMX
template<size_t N>
void xmrig::CpuWorker<N>::allocateRandomX_VM()
{
    RxDataset *dataset = Rx::dataset(m_job.currentJob(), m_node);

    while (dataset == nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        if (Nonce::sequence(Nonce::CPU) == 0) {
            return;
        }

        dataset = Rx::dataset(m_job.currentJob(), m_node);
    }

    if (!m_vm) {
        m_vm = RxVm::create(dataset, m_memory->scratchpad(), !m_hwAES, m_assembly, m_node);
    }
}
#endif


template<size_t N>
bool xmrig::CpuWorker<N>::selfTest()
{
#   ifdef XMRIG_ALGO_RANDOMX
    if (m_algorithm.family() == Algorithm::RANDOM_X) {
        return N == 1;
    }
#   endif

    allocateCnCtx();

    if (m_algorithm.family() == Algorithm::CN) {
        const bool rc = verify(Algorithm::CN_0,      test_output_v0)   &&
                        verify(Algorithm::CN_1,      test_output_v1)   &&
                        verify(Algorithm::CN_2,      test_output_v2)   &&
                        verify(Algorithm::CN_FAST,   test_output_msr)  &&
                        verify(Algorithm::CN_XAO,    test_output_xao)  &&
                        verify(Algorithm::CN_RTO,    test_output_rto)  &&
                        verify(Algorithm::CN_HALF,   test_output_half) &&
                        verify2(Algorithm::CN_R,     test_output_r)    &&
                        verify(Algorithm::CN_RWZ,    test_output_rwz)  &&
                        verify(Algorithm::CN_ZLS,    test_output_zls)  &&
                        verify(Algorithm::CN_CCX,    test_output_ccx)  &&
#                       ifdef XMRIG_ALGO_CN_GPU
                        verify(Algorithm::CN_GPU,    test_output_gpu)  &&
#                       endif
                        verify(Algorithm::CN_DOUBLE, test_output_double);

#       ifdef XMRIG_ALGO_CN_GPU
        if (!rc || N > 1) {
            return rc;
        }

        return verify(Algorithm::CN_GPU, test_output_gpu);
#       else
        return rc;
#       endif
    }

#   ifdef XMRIG_ALGO_CN_LITE
    if (m_algorithm.family() == Algorithm::CN_LITE) {
        return verify(Algorithm::CN_LITE_0,    test_output_v0_lite) &&
               verify(Algorithm::CN_LITE_1,    test_output_v1_lite);
    }
#   endif

#   ifdef XMRIG_ALGO_CN_HEAVY
    if (m_algorithm.family() == Algorithm::CN_HEAVY) {
        return verify(Algorithm::CN_HEAVY_0,    test_output_v0_heavy)  &&
               verify(Algorithm::CN_HEAVY_XHV,  test_output_xhv_heavy) &&
               verify(Algorithm::CN_HEAVY_TUBE, test_output_tube_heavy);
    }
#   endif

#   ifdef XMRIG_ALGO_CN_PICO
    if (m_algorithm.family() == Algorithm::CN_PICO) {
        return verify(Algorithm::CN_PICO_0, test_output_pico_trtl) &&
               verify(Algorithm::CN_PICO_TLO, test_output_pico_tlo);
    }
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    if (m_algorithm.family() == Algorithm::ARGON2) {
        return verify(Algorithm::AR2_CHUKWA, argon2_chukwa_test_out) &&
               verify(Algorithm::AR2_WRKZ, argon2_wrkz_test_out);
    }
#   endif

#   ifdef XMRIG_ALGO_ASTROBWT
    if (m_algorithm.family() == Algorithm::ASTROBWT) {
        return verify(Algorithm::ASTROBWT_DERO, astrobwt_dero_test_out);
    }
#   endif

    return false;
}


template<size_t N>
void xmrig::CpuWorker<N>::start()
{
    while (Nonce::sequence(Nonce::CPU) > 0) {
        if (Nonce::isPaused()) {
            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            while (Nonce::isPaused() && Nonce::sequence(Nonce::CPU) > 0);

            if (Nonce::sequence(Nonce::CPU) == 0) {
                break;
            }

            consumeJob();
        }

        uint64_t storeStatsMask = 7;

#       ifdef XMRIG_ALGO_RANDOMX
        bool first = true;
        uint64_t tempHash[8] = {};

        // RandomX is faster, we don't need to store stats so often
        if (m_job.currentJob().algorithm().family() == Algorithm::RANDOM_X) {
            storeStatsMask = 63;
        }
#       endif

        while (!Nonce::isOutdated(Nonce::CPU, m_job.sequence())) {
            if ((m_count & storeStatsMask) == 0) {
                storeStats();
            }

            const Job &job = m_job.currentJob();

            if (job.algorithm().l3() != m_algorithm.l3()) {
                break;
            }

            uint32_t current_job_nonces[N];
            for (size_t i = 0; i < N; ++i) {
                current_job_nonces[i] = *m_job.nonce(i);
            }

            bool valid = true;

#           ifdef XMRIG_ALGO_RANDOMX
            if (job.algorithm().family() == Algorithm::RANDOM_X) {

                if (first) {
                    first = false;
                    randomx_calculate_hash_first(m_vm, tempHash, m_job.blob(), job.size(), job.algorithm());
                }

                if (!nextRound(m_job)) {
                    break;
                }

                randomx_calculate_hash_next(m_vm, tempHash, m_job.blob(), job.size(), m_hash, job.algorithm());
            }
            else
#           endif
            {
#               ifdef XMRIG_ALGO_ASTROBWT
                if (job.algorithm().family() == Algorithm::ASTROBWT) {
                    if (!astrobwt::astrobwt_dero(m_job.blob(), job.size(), m_ctx[0]->memory, m_hash, m_astrobwtMaxSize, m_astrobwtAVX2))
                        valid = false;
                }
                else
#               endif
                {
                    fn(job.algorithm())(m_job.blob(), job.size(), m_hash, m_ctx, job.height());
                }

                if (!nextRound(m_job)) {
                    break;
                };
            }

            if (valid) {
                for (size_t i = 0; i < N; ++i) {
                    if (*reinterpret_cast<uint64_t*>(m_hash + (i * 32) + 24) < job.target()) {
                        JobResults::submit(job, current_job_nonces[i], m_hash + (i * 32));
                    }
                }
                m_count += N;
            }

            if (m_yield) {
                std::this_thread::yield();
            }
        }

        consumeJob();
    }
}


template<size_t N>
bool xmrig::CpuWorker<N>::verify(const Algorithm &algorithm, const uint8_t *referenceValue)
{
    cn_hash_fun func = fn(algorithm);
    if (!func) {
        return false;
    }

    func(test_input, 76, m_hash, m_ctx, 0);
    return memcmp(m_hash, referenceValue, sizeof m_hash) == 0;
}


template<size_t N>
bool xmrig::CpuWorker<N>::verify2(const Algorithm &algorithm, const uint8_t *referenceValue)
{
    cn_hash_fun func = fn(algorithm);
    if (!func) {
        return false;
    }

    for (size_t i = 0; i < (sizeof(cn_r_test_input) / sizeof(cn_r_test_input[0])); ++i) {
        const size_t size = cn_r_test_input[i].size;
        for (size_t k = 0; k < N; ++k) {
            memcpy(m_job.blob() + (k * size), cn_r_test_input[i].data, size);
        }

        func(m_job.blob(), size, m_hash, m_ctx, cn_r_test_input[i].height);

        for (size_t k = 0; k < N; ++k) {
            if (memcmp(m_hash + k * 32, referenceValue + i * 32, sizeof m_hash / N) != 0) {
                return false;
            }
        }
    }

    return true;
}


namespace xmrig {

template<>
bool CpuWorker<1>::verify2(const Algorithm &algorithm, const uint8_t *referenceValue)
{
    cn_hash_fun func = fn(algorithm);
    if (!func) {
        return false;
    }

    for (size_t i = 0; i < (sizeof(cn_r_test_input) / sizeof(cn_r_test_input[0])); ++i) {
        func(cn_r_test_input[i].data, cn_r_test_input[i].size, m_hash, m_ctx, cn_r_test_input[i].height);

        if (memcmp(m_hash, referenceValue + i * 32, sizeof m_hash) != 0) {
            return false;
        }
    }

    return true;
}

} // namespace xmrig


template<size_t N>
void xmrig::CpuWorker<N>::allocateCnCtx()
{
    if (m_ctx[0] == nullptr) {
        CnCtx::create(m_ctx, m_memory->scratchpad(), m_algorithm.l3(), N);
    }
}


template<size_t N>
void xmrig::CpuWorker<N>::consumeJob()
{
    if (Nonce::sequence(Nonce::CPU) == 0) {
        return;
    }

    m_job.add(m_miner->job(), kReserveCount, Nonce::CPU);

#   ifdef XMRIG_ALGO_RANDOMX
    if (m_job.currentJob().algorithm().family() == Algorithm::RANDOM_X) {
        allocateRandomX_VM();
    }
    else
#   endif
    {
        allocateCnCtx();
    }
}


namespace xmrig {

template class CpuWorker<1>;
template class CpuWorker<2>;
template class CpuWorker<3>;
template class CpuWorker<4>;
template class CpuWorker<5>;

} // namespace xmrig

