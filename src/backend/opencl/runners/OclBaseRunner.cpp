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


#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/runners/OclBaseRunner.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/net/stratum/Job.h"


xmrig::OclBaseRunner::OclBaseRunner(size_t, const OclLaunchData &data) :
    m_algorithm(data.algorithm),
    m_ctx(data.ctx)
{
    cl_int ret;
    m_queue = OclLib::createCommandQueue(m_ctx, data.device.id(), &ret);
    if (ret != CL_SUCCESS) {
        return;
    }

    m_input  = OclLib::createBuffer(m_ctx, CL_MEM_READ_ONLY, Job::kMaxBlobSize, nullptr, &ret);
    m_output = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * 0x100, nullptr, &ret);
}


xmrig::OclBaseRunner::~OclBaseRunner()
{
    OclLib::releaseMemObject(m_input);
    OclLib::releaseMemObject(m_output);

    OclLib::releaseCommandQueue(m_queue);
}


bool xmrig::OclBaseRunner::selfTest() const
{
    return m_queue != nullptr && m_input != nullptr && m_output != nullptr && !m_options.empty();
}



const char *xmrig::OclBaseRunner::buildOptions() const
{
    return m_options.c_str();
}


void xmrig::OclBaseRunner::run(uint32_t *hashOutput)
{

}


void xmrig::OclBaseRunner::set(const Job &job)
{

}
