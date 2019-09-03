/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#include <cstring>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>


#include "backend/opencl/cl/cn/cryptonight_r_cl.h"
#include "backend/opencl/interfaces/IOclRunner.h"
#include "backend/opencl/OclCache.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/OclThread.h"
#include "backend/opencl/runners/tools/OclCnR.h"
#include "backend/opencl/wrappers/OclError.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/tools/Chrono.h"
#include "crypto/cn/CryptoNight_monero.h"


namespace xmrig {


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


class CacheEntry
{
public:
    inline CacheEntry(const Algorithm &algorithm, uint64_t heightOffset, uint32_t deviceIndex, cl_program program) :
        algorithm(algorithm),
        program(program),
        deviceIndex(deviceIndex),
        heightOffset(heightOffset)
    {}

    const Algorithm algorithm;
    const cl_program program;
    const uint32_t deviceIndex;
    const uint64_t heightOffset;
};


static std::mutex mutex;
static std::vector<CacheEntry> cache;


static cl_program search(const Algorithm &algorithm, uint64_t offset, uint32_t index)
{
    std::lock_guard<std::mutex> lock(mutex);

    for (const CacheEntry &entry : cache) {
        if (entry.heightOffset == offset && entry.deviceIndex == index && entry.algorithm == algorithm) {
            return entry.program;
        }
    }

    return nullptr;
}


static inline cl_program search(const IOclRunner &runner, uint64_t offset) { return search(runner.algorithm(), offset, runner.data().thread.index()); }


cl_program build(const IOclRunner &runner, const std::string &source, uint64_t offset)
{
    std::lock_guard<std::mutex> lock(mutex);

    cl_int ret;
    cl_device_id device = runner.data().device.id();
    const char *s       = source.c_str();

    cl_program program = OclLib::createProgramWithSource(runner.ctx(), 1, &s, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        return nullptr;
    }

    if (OclLib::buildProgram(program, 1, &device, runner.buildOptions()) != CL_SUCCESS) {
        printf("BUILD LOG:\n%s\n", OclLib::getProgramBuildLog(program, device).data());

        OclLib::releaseProgram(program);
        return nullptr;
    }

    cache.emplace_back(runner.algorithm(), offset, runner.data().thread.index(), program);

    return program;
}


} // namespace xmrig



cl_program xmrig::OclCnR::get(const IOclRunner &runner, uint64_t height, bool background)
{
    const uint64_t offset = (height / kHeightChunkSize) * kHeightChunkSize;

    cl_program program = search(runner, offset);
    if (program) {
        return program;
    }

    std::string source(cryptonight_r_defines_cl);

    for (size_t i = 0; i < kHeightChunkSize; ++i) {
        V4_Instruction code[256];
        const int code_size      = v4_random_math_init<Algorithm::CN_R>(code, offset + i);
        const std::string kernel = std::regex_replace(cryptonight_r_cl, std::regex("XMRIG_INCLUDE_RANDOM_MATH"), getCode(code, code_size));

        source += std::regex_replace(kernel, std::regex("KERNEL_NAME"), "cn1_" + std::to_string(offset + i));
    }

    return build(runner, source, offset);;
}
