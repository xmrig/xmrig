/* XMRig
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "ghostrider.h"
#include "sph_blake.h"
#include "sph_bmw.h"
#include "sph_groestl.h"
#include "sph_jh.h"
#include "sph_keccak.h"
#include "sph_skein.h"
#include "sph_luffa.h"
#include "sph_cubehash.h"
#include "sph_shavite.h"
#include "sph_simd.h"
#include "sph_echo.h"
#include "sph_hamsi.h"
#include "sph_fugue.h"
#include "sph_shabal.h"
#include "sph_whirlpool.h"

#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/tools/Chrono.h"
#include "backend/cpu/Cpu.h"
#include "crypto/cn/CnHash.h"
#include "crypto/cn/CnCtx.h"
#include "crypto/cn/CryptoNight.h"
#include "crypto/common/VirtualMemory.h"

#include <thread>
#include <atomic>
#include <uv.h>

#ifdef XMRIG_FEATURE_HWLOC
#include "base/kernel/Platform.h"
#include "backend/cpu/platform/HwlocCpuInfo.h"
#include <hwloc.h>
#endif

#if defined(XMRIG_ARM)
#   include "crypto/cn/sse2neon.h"
#elif defined(__GNUC__)
#   include <x86intrin.h>
#else
#   include <intrin.h>
#endif

#define CORE_HASH(i, x) static void h##i(const uint8_t* data, size_t size, uint8_t* output) \
{ \
    sph_##x##_context ctx; \
    sph_##x##_init(&ctx); \
    sph_##x(&ctx, data, size); \
    sph_##x##_close(&ctx, output); \
}

CORE_HASH( 0, blake512   );
CORE_HASH( 1, bmw512     );
CORE_HASH( 2, groestl512 );
CORE_HASH( 3, jh512      );
CORE_HASH( 4, keccak512  );
CORE_HASH( 5, skein512   );
CORE_HASH( 6, luffa512   );
CORE_HASH( 7, cubehash512);
CORE_HASH( 8, shavite512 );
CORE_HASH( 9, simd512    );
CORE_HASH(10, echo512    );
CORE_HASH(11, hamsi512   );
CORE_HASH(12, fugue512   );
CORE_HASH(13, shabal512  );
CORE_HASH(14, whirlpool  );

#undef CORE_HASH

typedef void (*core_hash_func)(const uint8_t* data, size_t size, uint8_t* output);
static const core_hash_func core_hash[15] = { h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14 };

namespace xmrig
{


static constexpr Algorithm::Id cn_hash[6] = {
    Algorithm::CN_GR_0,
    Algorithm::CN_GR_1,
    Algorithm::CN_GR_2,
    Algorithm::CN_GR_3,
    Algorithm::CN_GR_4,
    Algorithm::CN_GR_5,
};

static constexpr const char* cn_names[6] = {
    "cn/dark (512 KB)",
    "cn/dark-lite (256 KB)",
    "cn/fast (2 MB)",
    "cn/lite (1 MB)",
    "cn/turtle (256 KB)",
    "cn/turtle-lite (128 KB)",
};

static constexpr size_t cn_sizes[6] = {
    Algorithm::l3(Algorithm::CN_GR_0),     // 512 KB
    Algorithm::l3(Algorithm::CN_GR_1) / 2, // 256 KB
    Algorithm::l3(Algorithm::CN_GR_2),     // 2 MB
    Algorithm::l3(Algorithm::CN_GR_3),     // 1 MB
    Algorithm::l3(Algorithm::CN_GR_4),     // 256 KB
    Algorithm::l3(Algorithm::CN_GR_5) / 2, // 128 KB
};

static constexpr CnHash::AlgoVariant av_hw_aes[5] = { CnHash::AV_SINGLE, CnHash::AV_SINGLE, CnHash::AV_DOUBLE, CnHash::AV_TRIPLE, CnHash::AV_QUAD };
static constexpr CnHash::AlgoVariant av_soft_aes[5] = { CnHash::AV_SINGLE_SOFT, CnHash::AV_SINGLE_SOFT, CnHash::AV_DOUBLE_SOFT, CnHash::AV_TRIPLE_SOFT, CnHash::AV_QUAD_SOFT };

template<size_t N>
static inline void select_indices(uint32_t (&indices)[N], const uint8_t* seed)
{
    bool selected[N] = {};

    uint32_t k = 0;
    for (uint32_t i = 0; i < 64; ++i) {
        const uint8_t index = ((seed[i / 2] >> ((i & 1) * 4)) & 0xF) % N;
        if (!selected[index]) {
            selected[index] = true;
            indices[k++] = index;
            if (k >= N) {
                return;
            }
        }
    }

    for (uint32_t i = 0; i < N; ++i) {
        if (!selected[i]) {
            indices[k++] = i;
        }
    }
}


namespace ghostrider
{


#ifdef XMRIG_FEATURE_HWLOC


static struct AlgoTune
{
    double hashrate = 0.0;
    uint32_t step = 1;
    uint32_t threads = 1;
} tuneDefault[6], tune8MB[6];


struct HelperThread
{
    HelperThread(hwloc_bitmap_t cpu_set, int priority, bool is8MB) : m_cpuSet(cpu_set), m_priority(priority), m_is8MB(is8MB)
    {
        uv_mutex_init(&m_mutex);
        uv_cond_init(&m_cond);

        m_thread = new std::thread(&HelperThread::run, this);
        do {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (!m_ready);
    }

    ~HelperThread()
    {
        uv_mutex_lock(&m_mutex);
        m_finished = true;
        uv_cond_signal(&m_cond);
        uv_mutex_unlock(&m_mutex);

        m_thread->join();
        delete m_thread;

        uv_mutex_destroy(&m_mutex);
        uv_cond_destroy(&m_cond);

        hwloc_bitmap_free(m_cpuSet);
    }

    struct TaskBase
    {
        virtual ~TaskBase() {}
        virtual void run() = 0;
    };

    template<typename T>
    struct Task : TaskBase
    {
        inline Task(T&& task) : m_task(std::move(task))
        {
            static_assert(sizeof(Task) <= 128, "Task struct is too large");
        }

        void run() override
        {
            m_task();
            this->~Task();
        }

        T m_task;
    };

    template<typename T>
    inline void launch_task(T&& task)
    {
        uv_mutex_lock(&m_mutex);
        new (&m_tasks[m_numTasks++]) Task<T>(std::move(task));
        uv_cond_signal(&m_cond);
        uv_mutex_unlock(&m_mutex);
    }

    inline void wait() const
    {
        while (m_numTasks) {
            _mm_pause();
        }
    }

    void run()
    {
        if (hwloc_bitmap_weight(m_cpuSet) > 0) {
            hwloc_topology_t topology = reinterpret_cast<HwlocCpuInfo*>(Cpu::info())->topology();
            if (hwloc_set_cpubind(topology, m_cpuSet, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT) < 0) {
                hwloc_set_cpubind(topology, m_cpuSet, HWLOC_CPUBIND_THREAD);
            }
        }

        Platform::setThreadPriority(m_priority);

        uv_mutex_lock(&m_mutex);
        m_ready = true;

        do {
            uv_cond_wait(&m_cond, &m_mutex);

            const uint32_t n = m_numTasks;
            if (n > 0) {
                for (uint32_t i = 0; i < n; ++i) {
                    reinterpret_cast<TaskBase*>(&m_tasks[i])->run();
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                m_numTasks = 0;
            }
        } while (!m_finished);

        uv_mutex_unlock(&m_mutex);
    }

    uv_mutex_t m_mutex;
    uv_cond_t m_cond;

    alignas(16) uint8_t m_tasks[4][128] = {};
    volatile uint32_t m_numTasks = 0;
    volatile bool m_ready = false;
    volatile bool m_finished = false;
    hwloc_bitmap_t m_cpuSet = {};
    int m_priority = -1;
    bool m_is8MB = false;

    std::thread* m_thread = nullptr;
};


void benchmark()
{
#ifndef XMRIG_ARM
    static std::atomic<int> done{ 0 };
    if (done.exchange(1)) {
        return;
    }

    std::thread t([]() {
        // Try to avoid CPU core 0 because many system threads use it and can interfere
        uint32_t thread_index1 = (Cpu::info()->threads() > 2) ? 2 : 0;

        hwloc_topology_t topology = reinterpret_cast<HwlocCpuInfo*>(Cpu::info())->topology();
        hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index(topology, thread_index1);
        hwloc_obj_t pu2;
        hwloc_get_closest_objs(topology, pu, &pu2, 1);
        uint32_t thread_index2 = pu2 ? pu2->os_index : thread_index1;

        if (thread_index2 < thread_index1) {
            std::swap(thread_index1, thread_index2);
        }

        Platform::setThreadAffinity(thread_index1);
        Platform::setThreadPriority(3);

        constexpr uint32_t N = 1U << 21;

        VirtualMemory::init(0, N);
        VirtualMemory* memory = new VirtualMemory(N * 8, true, false, false);

        // 2 MB cache per core by default
        size_t max_scratchpad_size = 1U << 21;

        if ((Cpu::info()->L3() >> 22) > Cpu::info()->cores()) {
            // At least 1 core can run with 8 MB cache
            max_scratchpad_size = 1U << 23;
        }
        else if ((Cpu::info()->L3() >> 22) >= Cpu::info()->cores()) {
            // All cores can run with 4 MB cache
            max_scratchpad_size = 1U << 22;
        }

        LOG_VERBOSE("Running GhostRider benchmark on logical CPUs %u and %u (max scratchpad size %zu MB, huge pages %s)", thread_index1, thread_index2, max_scratchpad_size >> 20, memory->isHugePages() ? "on" : "off");

        cryptonight_ctx* ctx[8];
        CnCtx::create(ctx, memory->scratchpad(), N, 8);

        const CnHash::AlgoVariant* av = Cpu::info()->hasAES() ? av_hw_aes : av_soft_aes;

        uint8_t buf[80];
        uint8_t hash[32 * 8];

        LOG_VERBOSE("%24s |  N  | Hashrate", "Algorithm");
        LOG_VERBOSE("-------------------------|-----|-------------");

        for (uint32_t algo = 0; algo < 6; ++algo) {
            for (uint64_t step : { 1, 2, 4}) {
                const size_t cur_scratchpad_size = cn_sizes[algo] * step;
                if (cur_scratchpad_size > max_scratchpad_size) {
                    continue;
                }

                auto f = CnHash::fn(cn_hash[algo], av[step], Assembly::AUTO);

                double start_time = Chrono::highResolutionMSecs();

                double min_dt = 1e10;
                for (uint32_t iter = 0;; ++iter) {
                    double t1 = Chrono::highResolutionMSecs();

                    // Stop after 15 milliseconds, but only if at least 10 iterations were done
                    if ((iter >= 10) && (t1 - start_time >= 15.0)) {
                        break;
                    }

                    f(buf, sizeof(buf), hash, ctx, 0);

                    const double dt = Chrono::highResolutionMSecs() - t1;
                    if (dt < min_dt) {
                        min_dt = dt;
                    }
                }

                const double hashrate = step * 1e3 / min_dt;
                LOG_VERBOSE("%24s | %" PRIu64 "x1 | %.2f h/s", cn_names[algo], step, hashrate);

                if (hashrate > tune8MB[algo].hashrate) {
                    tune8MB[algo].hashrate = hashrate;
                    tune8MB[algo].step = static_cast<uint32_t>(step);
                    tune8MB[algo].threads = 1;
                }

                if ((cur_scratchpad_size < (1U << 23)) && (hashrate > tuneDefault[algo].hashrate)) {
                    tuneDefault[algo].hashrate = hashrate;
                    tuneDefault[algo].step = static_cast<uint32_t>(step);
                    tuneDefault[algo].threads = 1;
                }
            }
        }

        hwloc_bitmap_t helper_set = hwloc_bitmap_alloc();
        hwloc_bitmap_set(helper_set, thread_index2);
        HelperThread* helper = new HelperThread(helper_set, 3, false);

        for (uint32_t algo = 0; algo < 6; ++algo) {
            for (uint64_t step : { 1, 2, 4}) {
                const size_t cur_scratchpad_size = cn_sizes[algo] * step * 2;
                if (cur_scratchpad_size > max_scratchpad_size) {
                    continue;
                }

                auto f = CnHash::fn(cn_hash[algo], av[step], Assembly::AUTO);

                double start_time = Chrono::highResolutionMSecs();

                double min_dt = 1e10;
                for (uint32_t iter = 0;; ++iter) {
                    double t1 = Chrono::highResolutionMSecs();

                    // Stop after 30 milliseconds, but only if at least 10 iterations were done
                    if ((iter >= 10) && (t1 - start_time >= 30.0)) {
                        break;
                    }

                    helper->launch_task([&f, &buf, &hash, &ctx, &step]() { f(buf, sizeof(buf), hash + step * 32, ctx + step, 0); });
                    f(buf, sizeof(buf), hash, ctx, 0);
                    helper->wait();

                    const double dt = Chrono::highResolutionMSecs() - t1;
                    if (dt < min_dt) {
                        min_dt = dt;
                    }
                }

                const double hashrate = step * 2e3 / min_dt * 1.0075;
                LOG_VERBOSE("%24s | %" PRIu64 "x2 | %.2f h/s", cn_names[algo], step, hashrate);

                if (hashrate > tune8MB[algo].hashrate) {
                    tune8MB[algo].hashrate = hashrate;
                    tune8MB[algo].step = static_cast<uint32_t>(step);
                    tune8MB[algo].threads = 2;
                }

                if ((cur_scratchpad_size < (1U << 23)) && (hashrate > tuneDefault[algo].hashrate)) {
                    tuneDefault[algo].hashrate = hashrate;
                    tuneDefault[algo].step = static_cast<uint32_t>(step);
                    tuneDefault[algo].threads = 2;
                }
            }
        }

        delete helper;

        CnCtx::release(ctx, 8);
        delete memory;
    });

    t.join();

    LOG_VERBOSE("---------------------------------------------");
    LOG_VERBOSE("|         GhostRider tuning results         |");
    LOG_VERBOSE("---------------------------------------------");

    for (int algo = 0; algo < 6; ++algo) {
        LOG_VERBOSE("%24s | %ux%u | %.2f h/s", cn_names[algo], tuneDefault[algo].step, tuneDefault[algo].threads, tuneDefault[algo].hashrate);
        if ((tune8MB[algo].step != tuneDefault[algo].step) || (tune8MB[algo].threads != tuneDefault[algo].threads)) {
            LOG_VERBOSE("%24s | %ux%u | %.2f h/s", cn_names[algo], tune8MB[algo].step, tune8MB[algo].threads, tune8MB[algo].hashrate);
        }
    }
#endif
}


template <typename func>
static inline bool findByType(hwloc_obj_t obj, hwloc_obj_type_t type, func lambda)
{
    for (size_t i = 0; i < obj->arity; i++) {
        if (obj->children[i]->type == type) {
            if (lambda(obj->children[i])) {
                return true;
            }
        }
        else {
            if (findByType(obj->children[i], type, lambda)) {
                return true;
            }
        }
    }
    return false;
}


HelperThread* create_helper_thread(int64_t cpu_index, int priority, const std::vector<int64_t>& affinities)
{
#ifndef XMRIG_ARM
    hwloc_bitmap_t helper_cpu_set = hwloc_bitmap_alloc();
    hwloc_bitmap_t main_threads_set = hwloc_bitmap_alloc();

    for (int64_t i : affinities) {
        if (i >= 0) {
            hwloc_bitmap_set(main_threads_set, i);
        }
    }

    if (cpu_index >= 0) {
        hwloc_topology_t topology = reinterpret_cast<HwlocCpuInfo*>(Cpu::info())->topology();
        hwloc_obj_t root = hwloc_get_root_obj(topology);

        bool is8MB = false;

        findByType(root, HWLOC_OBJ_L3CACHE, [cpu_index, &is8MB](hwloc_obj_t obj) {
            if (!hwloc_bitmap_isset(obj->cpuset, cpu_index)) {
                return false;
            }

            uint32_t num_cores = 0;
            findByType(obj, HWLOC_OBJ_CORE, [&num_cores](hwloc_obj_t) { ++num_cores; return false; });

            if ((obj->attr->cache.size >> 22) > num_cores) {
                uint32_t num_8MB_cores = (obj->attr->cache.size >> 22) - num_cores;

                is8MB = findByType(obj, HWLOC_OBJ_CORE, [cpu_index, &num_8MB_cores](hwloc_obj_t obj2) {
                    if (num_8MB_cores > 0) {
                        --num_8MB_cores;
                        if (hwloc_bitmap_isset(obj2->cpuset, cpu_index)) {
                            return true;
                        }
                    }
                    return false;
                });
            }
            return true;
        });

        for (auto obj_type : { HWLOC_OBJ_CORE, HWLOC_OBJ_L1CACHE, HWLOC_OBJ_L2CACHE, HWLOC_OBJ_L3CACHE }) {
            findByType(root, obj_type, [cpu_index, helper_cpu_set, main_threads_set](hwloc_obj_t obj) {
                const hwloc_cpuset_t& s = obj->cpuset;
                if (hwloc_bitmap_isset(s, cpu_index)) {
                    hwloc_bitmap_andnot(helper_cpu_set, s, main_threads_set);
                    if (hwloc_bitmap_weight(helper_cpu_set) > 0) {
                        return true;
                    }
                }
                return false;
            });

            if (hwloc_bitmap_weight(helper_cpu_set) > 0) {
                return new HelperThread(helper_cpu_set, priority, is8MB);
            }
        }
    }
#endif

    return nullptr;
}


void destroy_helper_thread(HelperThread* t)
{
    delete t;
}


void hash_octa(const uint8_t* data, size_t size, uint8_t* output, cryptonight_ctx** ctx, HelperThread* helper, bool verbose)
{
    enum { N = 8 };

    uint8_t* ctx_memory[N];
    for (size_t i = 0; i < N; ++i) {
        ctx_memory[i] = ctx[i]->memory;
    }

    // PrevBlockHash (GhostRider's seed) is stored in bytes [4; 36)
    uint32_t core_indices[15];
    select_indices(core_indices, data + 4);

    uint32_t cn_indices[6];
    select_indices(cn_indices, data + 4);

    if (verbose) {
        static uint32_t prev_indices[3];
        if (memcmp(cn_indices, prev_indices, sizeof(prev_indices)) != 0) {
            memcpy(prev_indices, cn_indices, sizeof(prev_indices));
            for (int i = 0; i < 3; ++i) {
                LOG_INFO("%s GhostRider algo %d: %s", Tags::cpu(), i + 1, cn_names[cn_indices[i]]);
            }
        }
    }

    const CnHash::AlgoVariant* av = Cpu::info()->hasAES() ? av_hw_aes : av_soft_aes;
    const AlgoTune* tune = (helper && helper->m_is8MB) ? tune8MB : tuneDefault;

    uint8_t tmp[64 * N];

    if (helper && (tune[cn_indices[0]].threads == 2) && (tune[cn_indices[1]].threads == 2) && (tune[cn_indices[2]].threads == 2)) {
        const size_t n = N / 2;

        helper->launch_task([n, av, data, size, &ctx_memory, ctx, &cn_indices, &core_indices, &tmp, output, tune]() {
            const uint8_t* input = data;
            size_t input_size = size;

            for (size_t part = 0; part < 3; ++part) {
                const AlgoTune& t = tune[cn_indices[part]];

                // Allocate scratchpads
                {
                    uint8_t* p = ctx_memory[4];

                    for (size_t i = n, k = 4; i < N; ++i) {
                        if ((i % t.step) == 0) {
                            k = 4;
                            p = ctx_memory[4];
                        }
                        else if (p - ctx_memory[k] >= (1 << 21)) {
                            ++k;
                            p = ctx_memory[k];
                        }
                        ctx[i]->memory = p;
                        p += cn_sizes[cn_indices[part]];
                    }
                }

                for (size_t i = 0; i < 5; ++i) {
                    for (size_t j = n; j < N; ++j) {
                        core_hash[core_indices[part * 5 + i]](input + j * input_size, input_size, tmp + j * 64);
                    }
                    input = tmp;
                    input_size = 64;
                }

                auto f = CnHash::fn(cn_hash[cn_indices[part]], av[t.step], Assembly::AUTO);
                for (size_t j = n; j < N; j += t.step) {
                    f(tmp + j * 64, 64, output + j * 32, ctx + n, 0);
                }

                for (size_t j = n; j < N; ++j) {
                    memcpy(tmp + j * 64, output + j * 32, 32);
                    memset(tmp + j * 64 + 32, 0, 32);
                }
            }
        });

        const uint8_t* input = data;
        size_t input_size = size;

        for (size_t part = 0; part < 3; ++part) {
            const AlgoTune& t = tune[cn_indices[part]];

            // Allocate scratchpads
            {
                uint8_t* p = ctx_memory[0];

                for (size_t i = 0, k = 0; i < n; ++i) {
                    if ((i % t.step) == 0) {
                        k = 0;
                        p = ctx_memory[0];
                    }
                    else if (p - ctx_memory[k] >= (1 << 21)) {
                        ++k;
                        p = ctx_memory[k];
                    }
                    ctx[i]->memory = p;
                    p += cn_sizes[cn_indices[part]];
                }
            }

            for (size_t i = 0; i < 5; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    core_hash[core_indices[part * 5 + i]](input + j * input_size, input_size, tmp + j * 64);
                }
                input = tmp;
                input_size = 64;
            }

            auto f = CnHash::fn(cn_hash[cn_indices[part]], av[t.step], Assembly::AUTO);
            for (size_t j = 0; j < n; j += t.step) {
                f(tmp + j * 64, 64, output + j * 32, ctx, 0);
            }

            for (size_t j = 0; j < n; ++j) {
                memcpy(tmp + j * 64, output + j * 32, 32);
                memset(tmp + j * 64 + 32, 0, 32);
            }
        }

        helper->wait();
    }
    else {
        for (size_t part = 0; part < 3; ++part) {
            const AlgoTune& t = tune[cn_indices[part]];

            // Allocate scratchpads
            {
                uint8_t* p = ctx_memory[0];
                const size_t n = N / t.threads;

                // Thread 1
                for (size_t i = 0, k = 0; i < n; ++i) {
                    if ((i % t.step) == 0) {
                        k = 0;
                        p = ctx_memory[0];
                    }
                    else if (p - ctx_memory[k] >= (1 << 21)) {
                        ++k;
                        p = ctx_memory[k];
                    }
                    ctx[i]->memory = p;
                    p += cn_sizes[cn_indices[part]];
                }

                // Thread 2
                for (size_t i = n, k = 4; i < N; ++i) {
                    if ((i % t.step) == 0) {
                        k = 4;
                        p = ctx_memory[4];
                    }
                    else if (p - ctx_memory[k] >= (1 << 21)) {
                        ++k;
                        p = ctx_memory[k];
                    }
                    ctx[i]->memory = p;
                    p += cn_sizes[cn_indices[part]];
                }
            }

            size_t n = N;

            if (helper && (t.threads == 2)) {
                n = N / 2;

                helper->launch_task([data, size, n, &cn_indices, &core_indices, part, &tmp, av, &t, output, ctx]() {
                    const uint8_t* input = data;
                    size_t input_size = size;

                    for (size_t i = 0; i < 5; ++i) {
                        for (size_t j = n; j < N; ++j) {
                            core_hash[core_indices[part * 5 + i]](input + j * input_size, input_size, tmp + j * 64);
                        }
                        input = tmp;
                        input_size = 64;
                    }

                    auto f = CnHash::fn(cn_hash[cn_indices[part]], av[t.step], Assembly::AUTO);
                    for (size_t j = n; j < N; j += t.step) {
                        f(tmp + j * 64, 64, output + j * 32, ctx + n, 0);
                    }

                    for (size_t j = n; j < N; ++j) {
                        memcpy(tmp + j * 64, output + j * 32, 32);
                        memset(tmp + j * 64 + 32, 0, 32);
                    }
                });
            }

            for (size_t i = 0; i < 5; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    core_hash[core_indices[part * 5 + i]](data + j * size, size, tmp + j * 64);
                }
                data = tmp;
                size = 64;
            }

            auto f = CnHash::fn(cn_hash[cn_indices[part]], av[t.step], Assembly::AUTO);
            for (size_t j = 0; j < n; j += t.step) {
                f(tmp + j * 64, 64, output + j * 32, ctx, 0);
            }

            for (size_t j = 0; j < n; ++j) {
                memcpy(tmp + j * 64, output + j * 32, 32);
                memset(tmp + j * 64 + 32, 0, 32);
            }

            if (helper && (t.threads == 2)) {
                helper->wait();
            }
        }
    }

    for (size_t i = 0; i < N; ++i) {
        ctx[i]->memory = ctx_memory[i];
    }
}


#else // XMRIG_FEATURE_HWLOC


void benchmark() {}
HelperThread* create_helper_thread(int64_t, int, const std::vector<int64_t>&) { return nullptr; }
void destroy_helper_thread(HelperThread*) {}


void hash_octa(const uint8_t* data, size_t size, uint8_t* output, cryptonight_ctx** ctx, HelperThread*, bool verbose)
{
    constexpr uint32_t N = 8;

    // PrevBlockHash (GhostRider's seed) is stored in bytes [4; 36)
    const uint8_t* seed = data + 4;

    uint32_t core_indices[15];
    select_indices(core_indices, seed);

    uint32_t cn_indices[6];
    select_indices(cn_indices, seed);

#ifdef XMRIG_ARM
    uint32_t step[6] = { 1, 1, 1, 1, 1, 1 };
#else
    uint32_t step[6] = { 4, 4, 1, 2, 4, 4 };
#endif

    if (verbose) {
        static uint32_t prev_indices[3];
        if (memcmp(cn_indices, prev_indices, sizeof(prev_indices)) != 0) {
            memcpy(prev_indices, cn_indices, sizeof(prev_indices));
            for (int i = 0; i < 3; ++i) {
                LOG_INFO("%s GhostRider algo %d: %s", Tags::cpu(), i + 1, cn_names[cn_indices[i]]);
            }
        }
    }

    const CnHash::AlgoVariant* av = Cpu::info()->hasAES() ? av_hw_aes : av_soft_aes;

    const cn_hash_fun f[3] = {
        CnHash::fn(cn_hash[cn_indices[0]], av[step[cn_indices[0]]], Assembly::AUTO),
        CnHash::fn(cn_hash[cn_indices[1]], av[step[cn_indices[1]]], Assembly::AUTO),
        CnHash::fn(cn_hash[cn_indices[2]], av[step[cn_indices[2]]], Assembly::AUTO),
    };

    uint8_t tmp[64 * N];

    for (uint64_t part = 0; part < 3; ++part) {
        for (uint64_t i = 0; i < 5; ++i) {
            for (uint64_t j = 0; j < N; ++j) {
                core_hash[core_indices[part * 5 + i]](data + j * size, size, tmp + j * 64);
                data = tmp;
                size = 64;
            }
        }
        for (uint64_t j = 0, k = step[cn_indices[part]]; j < N; j += k) {
            f[part](tmp + j * 64, 64, output + j * 32, ctx, 0);
        }
        for (uint64_t j = 0; j < N; ++j) {
            memcpy(tmp + j * 64, output + j * 32, 32);
            memset(tmp + j * 64 + 32, 0, 32);
        }
    }
}


#endif // XMRIG_FEATURE_HWLOC


} // namespace ghostrider


} // namespace xmrig
