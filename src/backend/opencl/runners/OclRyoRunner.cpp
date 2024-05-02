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


#include "backend/opencl/runners/OclRyoRunner.h"
#include "backend/opencl/kernels/Cn00RyoKernel.h"
#include "backend/opencl/kernels/Cn0Kernel.h"
#include "backend/opencl/kernels/Cn1RyoKernel.h"
#include "backend/opencl/kernels/Cn2RyoKernel.h"
#include "backend/opencl/kernels/CnBranchKernel.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "crypto/cn/CnAlgo.h"
#include <stdexcept>


xmrig::OclRyoRunner::OclRyoRunner(size_t index, const OclLaunchData &data) : OclBaseRunner(index, data)
{
    m_options += " -DITERATIONS="   + std::to_string(CnAlgo<>::iterations(m_algorithm)) + "U";
    m_options += " -DMASK="         + std::to_string(CnAlgo<>::mask(m_algorithm)) + "U";
    m_options += " -DWORKSIZE="     + std::to_string(data.thread.worksize()) + "U";
    m_options += " -DMEMORY="       + std::to_string(m_algorithm.l3()) + "LU";
    m_options += " -DCN_UNROLL="    + std::to_string(data.thread.unrollFactor());

    m_options += " -cl-fp32-correctly-rounded-divide-sqrt";
}


xmrig::OclRyoRunner::~OclRyoRunner()
{
    delete m_cn00;
    delete m_cn0;
    delete m_cn1;
    delete m_cn2;

    OclLib::release(m_scratchpads);
    OclLib::release(m_states);
}


size_t xmrig::OclRyoRunner::bufferSize() const
{
    return OclBaseRunner::bufferSize() + align(data().algorithm.l3() * m_intensity) + align(200 * m_intensity);
}


void xmrig::OclRyoRunner::run(uint32_t nonce, uint32_t *hashOutput)
{
    static const cl_uint zero = 0;

    const size_t w_size = data().thread.worksize();
    const size_t g_thd  = ((m_intensity + w_size - 1u) / w_size) * w_size;

    assert(g_thd % w_size == 0);

    enqueueWriteBuffer(m_output, CL_FALSE, sizeof(cl_uint) * 0xFF, sizeof(cl_uint), &zero);

    m_cn0->enqueue(m_queue, nonce, g_thd);
    m_cn00->enqueue(m_queue, g_thd);
    m_cn1->enqueue(m_queue, g_thd, w_size);
    m_cn2->enqueue(m_queue, nonce, g_thd);

    finalize(hashOutput);
}


void xmrig::OclRyoRunner::set(const Job &job, uint8_t *blob)
{
    if (job.size() > (Job::kMaxBlobSize - 4)) {
        throw std::length_error("job size too big");
    }

    blob[job.size()] = 0x01;
    memset(blob + job.size() + 1, 0, Job::kMaxBlobSize - job.size() - 1);

    enqueueWriteBuffer(m_input, CL_TRUE, 0, Job::kMaxBlobSize, blob);

    m_cn2->setTarget(job.target());
}


void xmrig::OclRyoRunner::build()
{
    OclBaseRunner::build();

    m_cn00 = new Cn00RyoKernel(m_program);
    m_cn00->setArgs(m_scratchpads, m_states);

    m_cn0 = new Cn0Kernel(m_program);
    m_cn0->setArgs(m_input, 0, m_scratchpads, m_states, m_intensity);

    m_cn1 = new Cn1RyoKernel(m_program);
    m_cn1->setArgs(m_scratchpads, m_states, m_intensity);

    m_cn2 = new Cn2RyoKernel(m_program);
    m_cn2->setArgs(m_scratchpads, m_states, m_output, m_intensity);
}


void xmrig::OclRyoRunner::init()
{
    OclBaseRunner::init();

    m_scratchpads = createSubBuffer(CL_MEM_READ_WRITE, data().algorithm.l3() * m_intensity);
    m_states      = createSubBuffer(CL_MEM_READ_WRITE, 200 * m_intensity);
}
