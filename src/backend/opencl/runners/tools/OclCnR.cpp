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

#include "backend/opencl/runners/tools/OclCnR.h"
#include "backend/opencl/cl/cn/cryptonight_r_cl.h"
#include "backend/opencl/interfaces/IOclRunner.h"
#include "backend/opencl/OclCache.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/OclThread.h"
#include "backend/opencl/wrappers/OclError.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/tools/Baton.h"
#include "base/tools/Chrono.h"
#include "crypto/cn/CryptoNight_monero.h"


#include <cstring>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <uv.h>


namespace xmrig {


class CnrCacheEntry
{
public:
    inline CnrCacheEntry(const Algorithm &algo, uint64_t offset, uint32_t index, cl_program program) :
        program(program),
        m_algo(algo),
        m_index(index),
        m_offset(offset)
    {}

    inline bool isExpired(uint64_t offset) const                                    { return m_offset + OclCnR::kHeightChunkSize < offset; }
    inline bool match(const Algorithm &algo, uint64_t offset, uint32_t index) const { return m_algo == algo && m_offset == offset && m_index == index; }
    inline bool match(const IOclRunner &runner, uint64_t offset) const              { return match(runner.algorithm(), offset, runner.deviceIndex()); }
    inline void release() const                                                     { OclLib::release(program); }

    cl_program program;

private:
    Algorithm m_algo;
    uint32_t m_index;
    uint64_t m_offset;
};


class CnrCache
{
public:
    CnrCache() = default;

    inline cl_program search(const IOclRunner &runner, uint64_t offset) { return search(runner.algorithm(), offset, runner.deviceIndex()); }


    inline cl_program search(const Algorithm &algo, uint64_t offset, uint32_t index)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        for (const auto &entry : m_data) {
            if (entry.match(algo, offset, index)) {
                return entry.program;
            }
        }

        return nullptr;
    }


    void add(const Algorithm &algo, uint64_t offset, uint32_t index, cl_program program)
    {
        if (search(algo, offset, index)) {
            OclLib::release(program);

            return;
        }

        std::lock_guard<std::mutex> lock(m_mutex);

        gc(offset);
        m_data.emplace_back(algo, offset, index, program);
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
    void gc(uint64_t offset)
    {
        for (size_t i = 0; i < m_data.size();) {
            auto &entry = m_data[i];

            if (entry.isExpired(offset)) {
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
    std::vector<CnrCacheEntry> m_data;
};


static CnrCache cache;


class CnrBuilder
{
public:
    CnrBuilder() = default;

    cl_program build(const IOclRunner &runner, uint64_t offset)
    {
    #   ifdef APP_DEBUG
        const uint64_t ts = Chrono::steadyMSecs();
    #   endif

        std::lock_guard<std::mutex> lock(m_mutex);
        cl_program program = cache.search(runner, offset);
        if (program) {
            return program;
        }

        cl_int ret = 0;
        const std::string source = getSource(offset);
        cl_device_id device      = runner.data().device.id();
        const char *s            = source.c_str();

        program = OclLib::createProgramWithSource(runner.ctx(), 1, &s, nullptr, &ret);
        if (ret != CL_SUCCESS) {
            return nullptr;
        }

        if (OclLib::buildProgram(program, 1, &device, runner.buildOptions()) != CL_SUCCESS) {
            printf("BUILD LOG:\n%s\n", OclLib::getProgramBuildLog(program, device).data());

            OclLib::release(program);
            return nullptr;
        }

        LOG_DEBUG(GREEN_BOLD("[ocl]") " programs for heights %" PRIu64 " - %" PRIu64 " compiled. (%" PRIu64 "ms)", offset, offset + OclCnR::kHeightChunkSize - 1, Chrono::steadyMSecs() - ts);

        cache.add(runner.algorithm(), offset, runner.deviceIndex(), program);

        return program;
    }

private:
    static std::string getCode(const V4_Instruction *code, int code_size)
    {
        std::stringstream s;

        for (int i = 0; i < code_size; ++i) {
            const V4_Instruction inst = code[i];

            const uint32_t a = inst.dst_index;
            const uint32_t b = inst.src_index;

            switch (inst.opcode)
            {
            case MUL:
                s << 'r' << a << "*=r" << b << ';';
                break;

            case ADD:
                s << 'r' << a << "+=r" << b << '+' << inst.C << "U;";
                break;

            case SUB:
                s << 'r' << a << "-=r" << b << ';';
                break;

            case ROR:
            case ROL:
                s << 'r' << a << "=rotate(r" << a << ((inst.opcode == ROR) ? ",ROT_BITS-r" : ",r") << b << ");";
                break;

            case XOR:
                s << 'r' << a << "^=r" << b << ';';
                break;
            }

            s << '\n';
        }

        return s.str();
    }


    static std::string getSource(uint64_t offset)
    {
        std::string source(cryptonight_r_defines_cl);

        for (size_t i = 0; i < OclCnR::kHeightChunkSize; ++i) {
            V4_Instruction code[256];
            const int code_size      = v4_random_math_init<Algorithm::CN_R>(code, offset + i);
            const std::string kernel = std::regex_replace(std::string(cryptonight_r_cl), std::regex("XMRIG_INCLUDE_RANDOM_MATH"), getCode(code, code_size));

            source += std::regex_replace(kernel, std::regex("KERNEL_NAME"), "cn1_" + std::to_string(offset + i));
        }

        return source;
    }


    std::mutex m_mutex;
};


class CnrBaton : public Baton<uv_work_t>
{
public:
    inline CnrBaton(const IOclRunner &runner, uint64_t offset) :
        runner(runner),
        offset(offset)
    {}

    const IOclRunner &runner;
    const uint64_t offset;
};


static CnrBuilder builder;
static std::mutex bg_mutex;


} // namespace xmrig



cl_program xmrig::OclCnR::get(const IOclRunner &runner, uint64_t height)
{
    const uint64_t offset = (height / kHeightChunkSize) * kHeightChunkSize;

    if (offset + kHeightChunkSize - height == 1) {
        auto baton = new CnrBaton(runner, offset + kHeightChunkSize);

        uv_queue_work(uv_default_loop(), &baton->req,
            [](uv_work_t *req) {
                auto baton = static_cast<CnrBaton*>(req->data);

                std::lock_guard<std::mutex> lock(bg_mutex);

                builder.build(baton->runner, baton->offset);
            },
            [](uv_work_t *req, int) { delete static_cast<CnrBaton*>(req->data); }
        );
    }

    cl_program program = cache.search(runner, offset);
    if (program) {
        return program;
    }

    return builder.build(runner, offset);;
}


void xmrig::OclCnR::clear()
{
    std::lock_guard<std::mutex> lock(bg_mutex);

    cache.clear();
}
