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

#include "backend/opencl/runners/tools/OclKawPow.h"
#include "3rdparty/libethash/data_sizes.h"
#include "3rdparty/libethash/ethash_internal.h"
#include "backend/opencl/cl/kawpow/kawpow_cl.h"
#include "backend/opencl/interfaces/IOclRunner.h"
#include "backend/opencl/OclCache.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/OclThread.h"
#include "backend/opencl/wrappers/OclError.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/tools/Baton.h"
#include "base/tools/Chrono.h"
#include "crypto/kawpow/KPHash.h"


#include <cstring>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <uv.h>


namespace xmrig {


class KawPowCacheEntry
{
public:
    inline KawPowCacheEntry(const Algorithm &algo, uint64_t period, uint32_t worksize, uint32_t index, cl_program program, cl_kernel kernel) :
        program(program),
        kernel(kernel),
        m_algo(algo),
        m_index(index),
        m_period(period),
        m_worksize(worksize)
    {}

    inline bool isExpired(uint64_t period) const                                                       { return m_period + 1 < period; }
    inline bool match(const Algorithm &algo, uint64_t period, uint32_t worksize, uint32_t index) const { return m_algo == algo && m_period == period && m_worksize == worksize && m_index == index; }
    inline bool match(const IOclRunner &runner, uint64_t period, uint32_t worksize) const              { return match(runner.algorithm(), period, worksize, runner.deviceIndex()); }
    inline void release() const                                                                        { OclLib::release(kernel); OclLib::release(program); }

    cl_program program;
    cl_kernel kernel;

private:
    Algorithm m_algo;
    uint32_t m_index;
    uint64_t m_period;
    uint32_t m_worksize;
};


class KawPowCache
{
public:
    KawPowCache() = default;

    inline cl_kernel search(const IOclRunner &runner, uint64_t period, uint32_t worksize) { return search(runner.algorithm(), period, worksize, runner.deviceIndex()); }


    inline cl_kernel search(const Algorithm &algo, uint64_t period, uint32_t worksize, uint32_t index)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        for (const auto &entry : m_data) {
            if (entry.match(algo, period, worksize, index)) {
                return entry.kernel;
            }
        }

        return nullptr;
    }


    void add(const Algorithm &algo, uint64_t period, uint32_t worksize, uint32_t index, cl_program program, cl_kernel kernel)
    {
        if (search(algo, period, worksize, index)) {
            OclLib::release(kernel);
            OclLib::release(program);
            return;
        }

        std::lock_guard<std::mutex> lock(m_mutex);

        gc(period);
        m_data.emplace_back(algo, period, worksize, index, program, kernel);
    }


    void clear()
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        for (auto &entry : m_data) {
            entry.release();
        }

        m_data.clear();
    }


private:
    void gc(uint64_t period)
    {
        for (size_t i = 0; i < m_data.size();) {
            auto& entry = m_data[i];

            if (entry.isExpired(period)) {
                entry.release();
                entry = m_data.back();
                m_data.pop_back();
            }
            else {
                ++i;
            }
        }
    }


    std::mutex m_mutex;
    std::vector<KawPowCacheEntry> m_data;
};


static KawPowCache cache;


#define rnd()       (kiss99(rnd_state))
#define mix_src()   ("mix[" + std::to_string(rnd() % KPHash::REGS) + "]")
#define mix_dst()   ("mix[" + std::to_string(mix_seq_dst[(mix_seq_dst_cnt++) % KPHash::REGS]) + "]")
#define mix_cache() ("mix[" + std::to_string(mix_seq_cache[(mix_seq_cache_cnt++) % KPHash::REGS]) + "]")

class KawPowBaton : public Baton<uv_work_t>
{
public:
    inline KawPowBaton(const IOclRunner& runner, uint64_t period, uint32_t worksize) :
        runner(runner),
        period(period),
        worksize(worksize)
    {}

    const IOclRunner& runner;
    const uint64_t period;
    const uint32_t worksize;
};


class KawPowBuilder
{
public:
    ~KawPowBuilder()
    {
        if (m_loop) {
            uv_async_send(&m_shutdownAsync);
            uv_thread_join(&m_loopThread);
            delete m_loop;
        }
    }

    void build_async(const IOclRunner& runner, uint64_t period, uint32_t worksize);

    cl_kernel build(const IOclRunner &runner, uint64_t period, uint32_t worksize)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        const uint64_t ts = Chrono::steadyMSecs();

        cl_kernel kernel = cache.search(runner, period, worksize);
        if (kernel) {
            return kernel;
        }

        cl_int ret = 0;
        const std::string source = getSource(period);
        cl_device_id device      = runner.data().device.id();
        const char *s            = source.c_str();

        cl_program program = OclLib::createProgramWithSource(runner.ctx(), 1, &s, nullptr, &ret);
        if (ret != CL_SUCCESS) {
            return nullptr;
        }

        std::string options = " -DPROGPOW_DAG_ELEMENTS=";

        const uint64_t epoch = (period * KPHash::PERIOD_LENGTH) / KPHash::EPOCH_LENGTH;
        const uint64_t dag_elements = dag_sizes[epoch] / 256;

        options += std::to_string(dag_elements);

        options += " -DGROUP_SIZE=";
        options += std::to_string(worksize);

        options += runner.buildOptions();

        if (OclLib::buildProgram(program, 1, &device, options.c_str()) != CL_SUCCESS) {
            printf("BUILD LOG:\n%s\n", OclLib::getProgramBuildLog(program, device).data());

            OclLib::release(program);
            return nullptr;
        }

        kernel = OclLib::createKernel(program, "progpow_search", &ret);
        if (ret != CL_SUCCESS) {
            OclLib::release(program);
            return nullptr;
        }

        LOG_INFO("%s " YELLOW("KawPow") " program for period " WHITE_BOLD("%" PRIu64) " compiled " BLACK_BOLD("(%" PRIu64 "ms)"), Tags::opencl(), period, Chrono::steadyMSecs() - ts);

        cache.add(runner.algorithm(), period, worksize, runner.deviceIndex(), program, kernel);

        return kernel;
    }


private:
    std::mutex m_mutex;

    typedef struct {
        uint32_t z, w, jsr, jcong;
    } kiss99_t;


    static std::string getSource(uint64_t prog_seed)
    {
        std::stringstream ret;

        uint32_t seed0 = static_cast<uint32_t>(prog_seed);
        uint32_t seed1 = static_cast<uint32_t>(prog_seed >> 32);

        kiss99_t rnd_state;
        uint32_t fnv_hash = 0x811c9dc5;
        rnd_state.z = fnv1a(fnv_hash, seed0);
        rnd_state.w = fnv1a(fnv_hash, seed1);
        rnd_state.jsr = fnv1a(fnv_hash, seed0);
        rnd_state.jcong = fnv1a(fnv_hash, seed1);

        // Create a random sequence of mix destinations and cache sources
        // Merge is a read-modify-write, guaranteeing every mix element is modified every loop
        // Guarantee no cache load is duplicated and can be optimized away
        int mix_seq_dst[KPHash::REGS];
        int mix_seq_cache[KPHash::REGS];
        int mix_seq_dst_cnt = 0;
        int mix_seq_cache_cnt = 0;

        for (uint32_t i = 0; i < KPHash::REGS; i++) {
            mix_seq_dst[i] = i;
            mix_seq_cache[i] = i;
        }

        for (int i = KPHash::REGS - 1; i > 0; i--) {
            int j = 0;
            j = rnd() % (i + 1);
            std::swap(mix_seq_dst[i], mix_seq_dst[j]);
            j = rnd() % (i + 1);
            std::swap(mix_seq_cache[i], mix_seq_cache[j]);
        }

        for (int i = 0; (i < KPHash::CNT_CACHE) || (i < KPHash::CNT_MATH); ++i) {
            if (i < KPHash::CNT_CACHE) {
                // Cached memory access
                // lanes access random locations
                std::string src = mix_cache();
                std::string dest = mix_dst();
                uint32_t r = rnd();
                ret << "offset = " << src << " % PROGPOW_CACHE_WORDS;\n";
                ret << "data = c_dag[offset];\n";
                ret << merge(dest, "data", r);
            }

            if (i < KPHash::CNT_MATH) {
                // Random Math
                // Generate 2 unique sources
                int src_rnd = rnd() % ((KPHash::REGS - 1) * KPHash::REGS);
                int src1 = src_rnd % KPHash::REGS; // 0 <= src1 < KPHash::REGS
                int src2 = src_rnd / KPHash::REGS; // 0 <= src2 < KPHash::REGS - 1

                if (src2 >= src1) {
                    ++src2; // src2 is now any reg other than src1
                }

                std::string src1_str = "mix[" + std::to_string(src1) + "]";
                std::string src2_str = "mix[" + std::to_string(src2) + "]";
                uint32_t r1 = rnd();
                std::string dest = mix_dst();
                uint32_t r2 = rnd();
                ret << math("data", src1_str, src2_str, r1);
                ret << merge(dest, "data", r2);
            }
        }

        std::string kernel = std::regex_replace(std::string(kawpow_cl), std::regex("XMRIG_INCLUDE_PROGPOW_RANDOM_MATH"), ret.str());
        ret.str(std::string());

        ret << merge("mix[0]", "data_dag.s[0]", rnd());

        constexpr size_t num_words_per_lane = 256 / (sizeof(uint32_t) * KPHash::LANES);
        for (size_t i = 1; i < num_words_per_lane; i++)
        {
            std::string dest = mix_dst();
            uint32_t    r = rnd();
            ret << merge(dest, "data_dag.s[" + std::to_string(i) + "]", r);
        }

        kernel = std::regex_replace(kernel, std::regex("XMRIG_INCLUDE_PROGPOW_DATA_LOADS"), ret.str());
        return kernel;
    }


    static std::string merge(const std::string& a, const std::string& b, uint32_t r)
    {
        switch (r % 4)
        {
        case 0:
            return a + " = (" + a + " * 33) + " + b + ";\n";
        case 1:
            return a + " = (" + a + " ^ " + b + ") * 33;\n";
        case 2:
            return a + " = ROTL32(" + a + ", " + std::to_string(((r >> 16) % 31) + 1) + ") ^ " + b + ";\n";
        case 3:
            return a + " = ROTR32(" + a + ", " + std::to_string(((r >> 16) % 31) + 1) + ") ^ " + b + ";\n";
        }
        return "#error\n";
    }


    static std::string math(const std::string& d, const std::string& a, const std::string& b, uint32_t r)
    {
        switch (r % 11)
        {
        case 0:
            return d + " = " + a + " + " + b + ";\n";
        case 1:
            return d + " = " + a + " * " + b + ";\n";
        case 2:
            return d + " = mul_hi(" + a + ", " + b + ");\n";
        case 3:
            return d + " = min(" + a + ", " + b + ");\n";
        case 4:
            return d + " = ROTL32(" + a + ", " + b + " % 32);\n";
        case 5:
            return d + " = ROTR32(" + a + ", " + b + " % 32);\n";
        case 6:
            return d + " = " + a + " & " + b + ";\n";
        case 7:
            return d + " = " + a + " | " + b + ";\n";
        case 8:
            return d + " = " + a + " ^ " + b + ";\n";
        case 9:
            return d + " = clz(" + a + ") + clz(" + b + ");\n";
        case 10:
            return d + " = popcount(" + a + ") + popcount(" + b + ");\n";
        }
        return "#error\n";
    }


    static uint32_t fnv1a(uint32_t& h, uint32_t d)
    {
        return h = (h ^ d) * 0x1000193;
    }

    static uint32_t kiss99(kiss99_t& st)
    {
        st.z = 36969 * (st.z & 65535) + (st.z >> 16);
        st.w = 18000 * (st.w & 65535) + (st.w >> 16);
        uint32_t MWC = ((st.z << 16) + st.w);
        st.jsr ^= (st.jsr << 17);
        st.jsr ^= (st.jsr >> 13);
        st.jsr ^= (st.jsr << 5);
        st.jcong = 69069 * st.jcong + 1234567;
        return ((MWC ^ st.jcong) + st.jsr);
    }

private:
    uv_loop_t* m_loop          = nullptr;
    uv_thread_t m_loopThread   = {};
    uv_async_t m_shutdownAsync = {};
    uv_async_t m_batonAsync    = {};

    std::vector<KawPowBaton> m_batons;

    static void loop(void* data)
    {
        KawPowBuilder* builder = static_cast<KawPowBuilder*>(data);
        uv_run(builder->m_loop, UV_RUN_DEFAULT);
        uv_loop_close(builder->m_loop);
    }
};


static KawPowBuilder builder;


void KawPowBuilder::build_async(const IOclRunner& runner, uint64_t period, uint32_t worksize)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_loop) {
        m_loop = new uv_loop_t{};
        uv_loop_init(m_loop);

        uv_async_init(m_loop, &m_shutdownAsync, [](uv_async_t* handle)
            {
                KawPowBuilder* builder = reinterpret_cast<KawPowBuilder*>(handle->data);
                uv_close(reinterpret_cast<uv_handle_t*>(&builder->m_shutdownAsync), nullptr);
                uv_close(reinterpret_cast<uv_handle_t*>(&builder->m_batonAsync), nullptr);
            });

        uv_async_init(m_loop, &m_batonAsync, [](uv_async_t* handle)
            {
                std::vector<KawPowBaton> batons;
                {
                    KawPowBuilder* b = reinterpret_cast<KawPowBuilder*>(handle->data);

                    std::lock_guard<std::mutex> lock(b->m_mutex);
                    batons = std::move(b->m_batons);
                }

                for (const KawPowBaton& baton : batons) {
                    builder.build(baton.runner, baton.period, baton.worksize);
                }
            });

        m_shutdownAsync.data = this;
        m_batonAsync.data = this;

        uv_thread_create(&m_loopThread, loop, this);
    }

    m_batons.emplace_back(runner, period, worksize);
    uv_async_send(&m_batonAsync);
}


cl_kernel OclKawPow::get(const IOclRunner &runner, uint64_t height, uint32_t worksize)
{
    const uint64_t period = height / KPHash::PERIOD_LENGTH;

    if (!cache.search(runner, period + 1, worksize)) {
        builder.build_async(runner, period + 1, worksize);
    }

    cl_kernel kernel = cache.search(runner, period, worksize);
    if (kernel) {
        return kernel;
    }

    return builder.build(runner, period, worksize);
}


void OclKawPow::clear()
{
    cache.clear();
}

} // namespace xmrig
