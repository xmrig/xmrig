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


#include "backend/opencl/cl/OclSource.h"
#include "backend/opencl/OclCache.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/runners/OclBaseRunner.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "backend/opencl/wrappers/OclError.h"


xmrig::OclBaseRunner::OclBaseRunner(size_t id, const OclLaunchData &data) :
    m_algorithm(data.algorithm),
    m_ctx(data.ctx),
    m_source(OclSource::get(data.algorithm)),
    m_data(data),
    m_threadId(id)
{
    m_deviceKey = data.device.name();

#   ifdef XMRIG_STRICT_OPENCL_CACHE
    m_deviceKey += ":";
    m_deviceKey += data.platform.version();

    m_deviceKey += ":";
    m_deviceKey += OclLib::getDeviceString(data.device.id(), CL_DRIVER_VERSION);
#   endif

#   if defined(__x86_64__) || defined(_M_AMD64) || defined (__arm64__) || defined (__aarch64__)
    m_deviceKey += ":64";
#   endif
}


xmrig::OclBaseRunner::~OclBaseRunner()
{
    OclLib::release(m_program);
    OclLib::release(m_input);
    OclLib::release(m_output);
    OclLib::release(m_queue);
}


uint32_t xmrig::OclBaseRunner::deviceIndex() const
{
    return data().thread.index();
}


void xmrig::OclBaseRunner::build()
{
    m_program = OclCache::build(this);

    if (m_program == nullptr) {
        throw std::runtime_error(OclError::toString(CL_INVALID_PROGRAM));
    }
}


void xmrig::OclBaseRunner::init()
{
    m_queue  = OclLib::createCommandQueue(m_ctx, data().device.id());
    m_input  = OclLib::createBuffer(m_ctx, CL_MEM_READ_ONLY, Job::kMaxBlobSize);
    m_output = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * 0x100);
}


void xmrig::OclBaseRunner::enqueueReadBuffer(cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr)
{
    const cl_int ret = OclLib::enqueueReadBuffer(m_queue, buffer, blocking_read, offset, size, ptr, 0, nullptr, nullptr);
    if (ret != CL_SUCCESS) {
        throw std::runtime_error(OclError::toString(ret));
    }
}


void xmrig::OclBaseRunner::enqueueWriteBuffer(cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr)
{
    const cl_int ret = OclLib::enqueueWriteBuffer(m_queue, buffer, blocking_write, offset, size, ptr, 0, nullptr, nullptr);
    if (ret != CL_SUCCESS) {
        throw std::runtime_error(OclError::toString(ret));
    }
}


void xmrig::OclBaseRunner::finalize(uint32_t *hashOutput)
{
    enqueueReadBuffer(m_output, CL_TRUE, 0, sizeof(cl_uint) * 0x100, hashOutput);

    uint32_t &results = hashOutput[0xFF];
    if (results > 0xFF) {
        results = 0xFF;
    }
}
