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

#include <cassert>
#include <thread>
#include <mutex>


#include "backend/cpu/Cpu.h"
#include "backend/cpu/CpuWorker.h"
#include "base/tools/Alignment.h"
#include "base/tools/Chrono.h"
#include "core/config/Config.h"
#include "core/Miner.h"
#include "crypto/cn/CnCtx.h"
#include "crypto/cn/CryptoNight_test.h"
#include "crypto/cn/CryptoNight.h"
#include "crypto/common/Nonce.h"
#include "crypto/common/VirtualMemory.h"
#include "crypto/rx/Rx.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"
#include "crypto/rx/RxVm.h"
#include "crypto/ghostrider/ghostrider.h"
#include "crypto/flex/flex.h"
#include "net/JobResults.h"


#ifdef XMRIG_ALGO_RANDOMX
#   include "crypto/randomx/randomx.h"
#endif


#ifdef XMRIG_FEATURE_BENCHMARK
#   include "backend/common/benchmark/BenchState.h"
#endif


namespace xmrig {

static constexpr uint32_t kReserveCount = 32768;


#ifdef XMRIG_ALGO_CN_HEAVY
static std::mutex cn_heavyZen3MemoryMutex;
VirtualMemory* cn_heavyZen3Memory = nullptr;
#endif

} // namespace xmrig



template<size_t N>
xmrig::CpuWorker<N>::CpuWorker(size_t id, const CpuLaunchData &data) :
    Worker(id, data.affinity, data.priority),
    m_algorithm(data.algorithm),
    m_assembly(data.assembly),
    m_hwAES(data.hwAES),
    m_yield(data.yield),
    m_av(data.av()),
    m_miner(data.miner),
    m_threads(data.threads),
    m_ctx()
{
#   ifdef XMRIG_ALGO_CN_HEAVY
    // cn-heavy optimization for Zen3 CPUs
    const auto arch = Cpu::info()->arch();
    const uint32_t model = Cpu::info()->model();
    const bool is_vermeer = (arch == ICpuInfo::ARCH_ZEN3) && (model == 0x21);
    const bool is_raphael = (arch == ICpuInfo::ARCH_ZEN4) && (model == 0x61);
    if ((N == 1) && (m_av == CnHash::AV_SINGLE) && (m_algorithm.family() == Algorithm::CN_HEAVY) && (m_assembly != Assembly::NONE) && (is_vermeer || is_raphael)) {
        std::lock_guard<std::mutex> lock(cn_heavyZen3MemoryMutex);
        if (!cn_heavyZen3Memory) {
            // Round up number of threads to the multiple of 8
            const size_t num_threads = ((m_threads + 7) / 8) * 8;
            cn_heavyZen3Memory = new VirtualMemory(m_algorithm.l3() * num_threads, data.hugePages, false, false, node());
        }
        m_memory = cn_heavyZen3Memory;
    }
    else
#   endif
    {
        m_memory = new VirtualMemory(m_algorithm.l3() * N, data.hugePages, false, true, node());
    }

#   ifdef XMRIG_ALGO_GHOSTRIDER
    m_ghHelper = ghostrider::create_helper_thread(affinity(), data.priority, data.affinities);
#   endif
}


template<size_t N>
xmrig::CpuWorker<N>::~CpuWorker()
{
#   ifdef XMRIG_ALGO_RANDOMX
    RxVm::destroy(m_vm);
#   endif

    CnCtx::release(m_ctx, N);

#   ifdef XMRIG_ALGO_CN_HEAVY
    if (m_memory != cn_heavyZen3Memory)
#   endif
    {
        delete m_memory;
    }

#   ifdef XMRIG_ALGO_GHOSTRIDER
    ghostrider::destroy_helper_thread(m_ghHelper);
#   endif
}


#ifdef XMRIG_ALGO_RANDOMX
template<size_t N>
void xmrig::CpuWorker<N>::allocateRandomX_VM()
{
    RxDataset *dataset = Rx::dataset(m_job.currentJob(), node());

    while (dataset == nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        if (Nonce::sequence(Nonce::CPU) == 0) {
            return;
        }

        dataset = Rx::dataset(m_job.currentJob(), node());
    }

    if (!m_vm) {
        // Try to allocate scratchpad from dataset's 1 GB huge pages, if normal huge pages are not available
        uint8_t* scratchpad = m_memory->isHugePages() ? m_memory->scratchpad() : dataset->tryAllocateScrathpad();
        m_vm = RxVm::create(dataset, scratchpad ? scratchpad : m_memory->scratchpad(), !m_hwAES, m_assembly, node());
    }
    else if (!dataset->get() && (m_job.currentJob().seed() != m_seed)) {
        // Update RandomX light VM with the new seed
        randomx_vm_set_cache(m_vm, dataset->cache()->get());
    }
    m_seed = m_job.currentJob().seed();
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

#   ifdef XMRIG_ALGO_GHOSTRIDER
    if (m_algorithm.family() == Algorithm::GHOSTRIDER) {
        switch (m_algorithm.id()) {
            case Algorithm::GHOSTRIDER_RTM:
                return (N == 8) && verify(Algorithm::GHOSTRIDER_RTM, test_output_gr);
            case Algorithm::FLEX_KCN:
                return (N == 1) && verify(Algorithm::FLEX_KCN, test_output_flex);
            default:;
        }
    }
#   endif

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
                        verify(Algorithm::CN_DOUBLE, test_output_double)
#                       ifdef XMRIG_ALGO_CN_GPU
                        &&
                        verify(Algorithm::CN_GPU,    test_output_gpu)
#                       endif
                        ;

#       ifdef XMRIG_ALGO_CN_GPU
        if (! (!rc || N > 1)) {
            return verify(Algorithm::CN_GPU, test_output_gpu);
        } else
#       endif
        return rc;
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

#   ifdef XMRIG_ALGO_CN_FEMTO
    if (m_algorithm.family() == Algorithm::CN_FEMTO) {
        return verify(Algorithm::CN_UPX2, test_output_femto_upx2);
    }
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    if (m_algorithm.family() == Algorithm::ARGON2) {
        return verify(Algorithm::AR2_CHUKWA, argon2_chukwa_test_out) &&
               verify(Algorithm::AR2_CHUKWA_V2, argon2_chukwa_v2_test_out) &&
               verify(Algorithm::AR2_WRKZ, argon2_wrkz_test_out);
    }
#   endif

    return false;
}


template<size_t N>
void xmrig::CpuWorker<N>::hashrateData(uint64_t &hashCount, uint64_t &, uint64_t &rawHashes) const
{
    hashCount = m_count;
    rawHashes = m_count;
}


template<size_t N>
void xmrig::CpuWorker<N>::start()
{
    while (Nonce::sequence(Nonce::CPU) > 0) {
        if (Nonce::isPaused()) {
            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            while (Nonce::isPaused() && Nonce::sequence(Nonce::CPU) > 0);

            if (Nonce::sequence(Nonce::CPU) == 0) {
                break;
            }

            consumeJob();
        }

#       ifdef XMRIG_ALGO_RANDOMX
        bool first = true;
        alignas(16) uint64_t tempHash[8] = {};
#       endif

        while (!Nonce::isOutdated(Nonce::CPU, m_job.sequence())) {
            const Job &job = m_job.currentJob();

            if (job.algorithm().l3() != m_algorithm.l3()) {
                break;
            }

            uint32_t current_job_nonces[N];
            for (size_t i = 0; i < N; ++i) {
                current_job_nonces[i] = readUnaligned(m_job.nonce(i));
            }

#           ifdef XMRIG_FEATURE_BENCHMARK
            if (m_benchSize) {
                if (current_job_nonces[0] >= m_benchSize) {
                    return BenchState::done();
                }

                // Make each hash dependent on the previous one in single thread benchmark to prevent cheating with multiple threads
                if (m_threads == 1) {
                    *(uint64_t*)(m_job.blob()) ^= BenchState::data();
                }
            }
#           endif

            bool valid = true;

            uint8_t miner_signature_saved[64];

#           ifdef XMRIG_ALGO_RANDOMX
            uint8_t* miner_signature_ptr = m_job.blob() + m_job.nonceOffset() + m_job.nonceSize();
            if (job.algorithm().family() == Algorithm::RANDOM_X) {

                if (first) {
                    first = false;
                    if (job.hasMinerSignature()) {
                        job.generateMinerSignature(m_job.blob(), job.size(), miner_signature_ptr);
                    }
                    randomx_calculate_hash_first(m_vm, tempHash, m_job.blob(), job.size(), job.algorithm());
                }

                if (!nextRound()) {
                    break;
                }

                if (job.hasMinerSignature()) {
                    memcpy(miner_signature_saved, miner_signature_ptr, sizeof(miner_signature_saved));
                    job.generateMinerSignature(m_job.blob(), job.size(), miner_signature_ptr);
                }
                randomx_calculate_hash_next(m_vm, tempHash, m_job.blob(), job.size(), m_hash, job.algorithm());
            }
            else
#           endif
            {
                switch (job.algorithm().family()) {

#               ifdef XMRIG_ALGO_GHOSTRIDER
                case Algorithm::GHOSTRIDER:
                    switch (job.algorithm()) {
                        case Algorithm::GHOSTRIDER_RTM:
                            if (N == 8) {
                                ghostrider::hash_octa(m_job.blob(), job.size(), m_hash, m_ctx, m_ghHelper);
                            } else {
                                valid = false;
                            }
                            break;
                        case Algorithm::FLEX_KCN:
                            if (N == 1) {
                                flex_hash(reinterpret_cast<const char*>(m_job.blob()), reinterpret_cast<char*>(m_hash), m_ctx);
                            } else {
                                valid = false;
                            }
                            break;
                        default:
                            valid = false;
                    }
                    break;
#               endif

                default:
                    fn(job.algorithm())(m_job.blob(), job.size(), m_hash, m_ctx, job.height());
                    break;
                }

                if (!nextRound()) {
                    break;
                };
            }

            if (valid) {
                for (size_t i = 0; i < N; ++i) {
                    const uint64_t value = *reinterpret_cast<uint64_t*>(m_hash + (i * 32) + 24);

#                   ifdef XMRIG_FEATURE_BENCHMARK
                    if (m_benchSize) {
                        if (current_job_nonces[i] < m_benchSize) {
                            BenchState::add(value);
                        }
                    }
                    else
#                   endif
                    if (value < job.target()) {
                        JobResults::submit(job, current_job_nonces[i], m_hash + (i * 32), job.hasMinerSignature() ? miner_signature_saved : nullptr);
                    }
                }
                m_count += N;
            }

            if (m_yield) {
                std::this_thread::yield();
            }
        }

        if (!Nonce::isPaused()) {
            consumeJob();
        }
    }
}


template<size_t N>
bool xmrig::CpuWorker<N>::nextRound()
{
#   ifdef XMRIG_FEATURE_BENCHMARK
    const uint32_t count = m_benchSize ? 1U : kReserveCount;
#   else
    constexpr uint32_t count = kReserveCount;
#   endif

    if (!m_job.nextRound(count, 1)) {
        JobResults::done(m_job.currentJob());

        return false;
    }

    return true;
}


template<size_t N>
bool xmrig::CpuWorker<N>::verify(const Algorithm &algorithm, const uint8_t *referenceValue)
{
#   ifdef XMRIG_ALGO_GHOSTRIDER
    switch (algorithm) {
      case Algorithm::GHOSTRIDER_RTM: {
        uint8_t blob[N * 80] = {};
        for (size_t i = 0; i < N; ++i) {
            blob[i * 80 + 0] = static_cast<uint8_t>(i);
            blob[i * 80 + 4] = 0x10;
            blob[i * 80 + 5] = 0x02;
        }

        uint8_t hash1[N * 32] = {};
        ghostrider::hash_octa(blob, 80, hash1, m_ctx, 0, false);

        for (size_t i = 0; i < N; ++i) {
            blob[i * 80 + 0] = static_cast<uint8_t>(i);
            blob[i * 80 + 4] = 0x43;
            blob[i * 80 + 5] = 0x05;
        }

        uint8_t hash2[N * 32] = {};
        ghostrider::hash_octa(blob, 80, hash2, m_ctx, 0, false);

        for (size_t i = 0; i < N * 32; ++i) {
            if ((hash1[i] ^ hash2[i]) != referenceValue[i]) {
                return false;
            }
        }

        return true;
      }
      case Algorithm::FLEX_KCN: {
        const uint8_t header[80] = {
          0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1c, 0xcc, 0xa6, 0x6a,
          0x44, 0xf8, 0xbd, 0x55, 0x45, 0xc3, 0x16, 0x4a, 0x3a, 0x76, 0xda, 0x50, 0x39, 0x53, 0x28, 0xc9, 0x07, 0x56, 0x33, 0x77,
          0x5b, 0xc4, 0xc8, 0x79, 0x8f, 0xd6, 0x77, 0x2b, 0x70, 0x0d, 0x21, 0x5c, 0xf0, 0xff, 0x0f, 0x1e, 0x00, 0x00, 0x00, 0x00
        };
        char hash[32] = {};
        flex_hash(reinterpret_cast<const char*>(header), hash, m_ctx);
        return memcmp(referenceValue, hash, sizeof hash) == 0;
      }
      default:;
    }
#   endif

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
        int shift = 0;

#       ifdef XMRIG_ALGO_CN_HEAVY
        // cn-heavy optimization for Zen3 CPUs
        if (m_memory == cn_heavyZen3Memory) {
            shift = (id() / 8) * m_algorithm.l3() * 8 + (id() % 8) * 64;
        }
#       endif

        CnCtx::create(m_ctx, m_memory->scratchpad() + shift, m_algorithm.l3(), N);
    }
}


template<size_t N>
void xmrig::CpuWorker<N>::consumeJob()
{
    if (Nonce::sequence(Nonce::CPU) == 0) {
        return;
    }

    auto job = m_miner->job();

#   ifdef XMRIG_FEATURE_BENCHMARK
    m_benchSize          = job.benchSize();
    const uint32_t count = m_benchSize ? 1U : kReserveCount;
#   else
    constexpr uint32_t count = kReserveCount;
#   endif

    m_job.add(job, count, Nonce::CPU);

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
template class CpuWorker<8>;

} // namespace xmrig

