/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include <fstream>
#include <map>
#include <mutex>
#include <sstream>


#include "backend/opencl/OclCache.h"
#include "3rdparty/base32/base32.h"
#include "backend/common/Tags.h"
#include "backend/opencl/interfaces/IOclRunner.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/crypto/keccak.h"
#include "base/io/log/Log.h"
#include "base/tools/Chrono.h"


namespace xmrig {


static std::mutex mutex;


static cl_program createFromSource(const IOclRunner *runner)
{
    LOG_INFO("%s GPU " WHITE_BOLD("#%zu") " " YELLOW_BOLD("compiling..."), ocl_tag(), runner->data().device.index());

    cl_int ret;
    cl_device_id device = runner->data().device.id();
    const char *source  = runner->source();
    const uint64_t ts   = Chrono::steadyMSecs();

    cl_program program = OclLib::createProgramWithSource(runner->ctx(), 1, &source, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        return nullptr;
    }

    if (OclLib::buildProgram(program, 1, &device, runner->buildOptions()) != CL_SUCCESS) {
        printf("BUILD LOG:\n%s\n", OclLib::getProgramBuildLog(program, device).data());

        OclLib::release(program);
        return nullptr;
    }

    LOG_INFO("%s GPU " WHITE_BOLD("#%zu") " " GREEN_BOLD("compilation completed") BLACK_BOLD(" (%" PRIu64 " ms)"),
             ocl_tag(), runner->data().device.index(), Chrono::steadyMSecs() - ts);

    return program;
}


static cl_program createFromBinary(const IOclRunner *runner, const std::string &fileName)
{
    std::ifstream file(fileName, std::ofstream::in | std::ofstream::binary);
    if (!file.good()) {
        return nullptr;
    }

    std::ostringstream ss;
    ss << file.rdbuf();

    const std::string s     = ss.str();
    const size_t bin_size   = s.size();
    auto data_ptr           = s.data();
    cl_device_id device     = runner->data().device.id();

    cl_int clStatus;
    cl_int ret;
    cl_program program = OclLib::createProgramWithBinary(runner->ctx(), 1, &device, &bin_size, reinterpret_cast<const unsigned char **>(&data_ptr), &clStatus, &ret);
    if (ret != CL_SUCCESS) {
        return nullptr;
    }

    if (OclLib::buildProgram(program, 1, &device) != CL_SUCCESS) {
        OclLib::release(program);
        return nullptr;
    }

    return program;
}


} // namespace xmrig


cl_program xmrig::OclCache::build(const IOclRunner *runner)
{
    std::lock_guard<std::mutex> lock(mutex);

    if (Nonce::sequence(Nonce::OPENCL) == 0) {
        return nullptr;
    }

    std::string fileName;
    if (runner->data().cache) {
#       ifdef _WIN32
        fileName = prefix() + "\\xmrig\\.cache\\" + cacheKey(runner) + ".bin";
#       else
        fileName = prefix() + "/.cache/" + cacheKey(runner) + ".bin";
#       endif

        cl_program program = createFromBinary(runner, fileName);
        if (program) {
            return program;
        }
    }

    cl_program program = createFromSource(runner);
    if (runner->data().cache && program) {
        save(program, fileName);
    }

    return program;
}


std::string xmrig::OclCache::cacheKey(const char *deviceKey, const char *options, const char *source)
{
    std::string in(source);
    in += options;
    in += deviceKey;

    uint8_t hash[200];
    keccak(in.c_str(), in.size(), hash);

    uint8_t result[32] = { 0 };
    base32_encode(hash, 12, result, sizeof(result));

    return reinterpret_cast<char *>(result);
}


std::string xmrig::OclCache::cacheKey(const IOclRunner *runner)
{
    return cacheKey(runner->deviceKey(), runner->buildOptions(), runner->source());
}


void xmrig::OclCache::save(cl_program program, const std::string &fileName)
{
    size_t size = 0;
    if (OclLib::getProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size), &size) != CL_SUCCESS) {
        return;
    }

    std::vector<char> binary(size);

    char *data = binary.data();
    if (OclLib::getProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(char *), &data) != CL_SUCCESS) {
        return;
    }

    createDirectory();

    std::ofstream file_stream;
    file_stream.open(fileName, std::ofstream::out | std::ofstream::binary);
    file_stream.write(binary.data(), static_cast<int64_t>(binary.size()));
    file_stream.close();
}
