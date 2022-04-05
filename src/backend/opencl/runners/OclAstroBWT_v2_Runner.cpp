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

#include "backend/opencl/runners/OclAstroBWT_v2_Runner.h"
#include "backend/opencl/kernels/astrobwt_v2/AstroBWT_v2_FindSharesKernel.h"
#include "backend/opencl/kernels/astrobwt_v2/AstroBWT_v2_BWT_FixOrderKernel.h"
#include "backend/opencl/kernels/astrobwt_v2/AstroBWT_v2_BWT_PreprocessKernel.h"
#include "backend/opencl/kernels/astrobwt_v2/AstroBWT_v2_Salsa20Kernel.h"
#include "backend/opencl/kernels/astrobwt_v2/AstroBWT_v2_SHA3InitialKernel.h"
#include "backend/opencl/kernels/astrobwt_v2/AstroBWT_v2_SHA3Kernel.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"


xmrig::OclAstroBWT_v2_Runner::OclAstroBWT_v2_Runner(size_t index, const OclLaunchData &data) : OclBaseRunner(index, data)
{
    switch (data.device.type())
    {
    case OclDevice::Baffin:
    case OclDevice::Ellesmere:
    case OclDevice::Polaris:
    case OclDevice::Lexa:
    case OclDevice::Vega_10:
    case OclDevice::Vega_20:
    case OclDevice::Raven:
        m_workgroup_size = 64;
        break;

    default:
        m_workgroup_size = 32;
        break;
    }

    m_options += " -DSALSA20_GROUP_SIZE=" + std::to_string(m_workgroup_size);
    m_bwt_allocation_size = m_intensity * BWT_DATA_STRIDE;
}


xmrig::OclAstroBWT_v2_Runner::~OclAstroBWT_v2_Runner()
{
    delete m_find_shares_kernel;
    delete m_bwt_fix_order_kernel;
    delete m_bwt_preprocess_kernel;
    delete m_salsa20_kernel;
    delete m_sha3_initial_kernel;
    delete m_sha3_kernel;

    OclLib::release(m_input);
    OclLib::release(m_hashes);
    OclLib::release(m_data);
    OclLib::release(m_keys);
    OclLib::release(m_temp_storage);
}


size_t xmrig::OclAstroBWT_v2_Runner::bufferSize() const
{
    return OclBaseRunner::bufferSize() +
        align(m_intensity * 32) +                      // m_hashes
        align(m_bwt_allocation_size) +                 // m_data
        align(m_bwt_allocation_size * 4) +             // m_keys
        align(m_bwt_allocation_size * 2);              // m_temp_storage
}


void xmrig::OclAstroBWT_v2_Runner::run(uint32_t nonce, uint32_t *hashOutput)
{
    const uint32_t zero = 0;
    enqueueWriteBuffer(m_output, CL_FALSE, sizeof(cl_uint) * 0xFF, sizeof(uint32_t), &zero);

    m_sha3_initial_kernel->setArg(2, sizeof(nonce), &nonce);
    m_sha3_initial_kernel->enqueue(m_queue, m_intensity);

    m_salsa20_kernel->enqueue(m_queue, m_intensity, m_workgroup_size);

    m_bwt_preprocess_kernel->enqueue(m_queue, m_intensity, 1024);
    m_bwt_fix_order_kernel->enqueue(m_queue, m_intensity, 1024);

    m_sha3_kernel->enqueue(m_queue, m_intensity);

    m_find_shares_kernel->enqueue(m_queue, m_intensity, m_workgroup_size);

    finalize(hashOutput);

    OclLib::finish(m_queue);

    for (uint32_t i = 0; i < hashOutput[0xFF]; ++i) {
        hashOutput[i] += nonce;
    }
}


void xmrig::OclAstroBWT_v2_Runner::set(const Job &job, uint8_t *blob)
{
    if (job.size() > (Job::kMaxBlobSize - 4)) {
        throw std::length_error("job size too big");
    }

    if (job.size() < Job::kMaxBlobSize) {
        memset(blob + job.size(), 0, Job::kMaxBlobSize - job.size());
    }

    enqueueWriteBuffer(m_input, CL_TRUE, 0, Job::kMaxBlobSize, blob);

    m_sha3_initial_kernel->setArgs(m_input, static_cast<uint32_t>(job.size()), *job.nonce(), m_hashes);
    m_salsa20_kernel->setArgs(m_hashes, m_data);
    m_bwt_preprocess_kernel->setArgs(m_data, m_keys);
    m_bwt_fix_order_kernel->setArgs(m_data, m_keys, m_temp_storage);
    m_sha3_kernel->setArgs(m_temp_storage, m_hashes);
    m_find_shares_kernel->setArgs(m_hashes, m_output);
    m_find_shares_kernel->setTarget(job.target());
}


void xmrig::OclAstroBWT_v2_Runner::build()
{
    OclBaseRunner::build();

    m_find_shares_kernel = new AstroBWT_v2_FindSharesKernel(m_program);
    m_bwt_fix_order_kernel = new AstroBWT_v2_BWT_FixOrderKernel(m_program);
    m_bwt_preprocess_kernel = new AstroBWT_v2_BWT_PreprocessKernel(m_program);
    m_salsa20_kernel = new AstroBWT_v2_Salsa20Kernel(m_program);
    m_sha3_initial_kernel = new AstroBWT_v2_SHA3InitialKernel(m_program);
    m_sha3_kernel = new AstroBWT_v2_SHA3Kernel(m_program);
}


void xmrig::OclAstroBWT_v2_Runner::init()
{
    OclBaseRunner::init();

    const cl_mem_flags f = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

    m_hashes       = createSubBuffer(f, m_intensity * 32);
    m_data         = createSubBuffer(f, m_bwt_allocation_size);
    m_keys         = createSubBuffer(f, m_bwt_allocation_size * 4);
    m_temp_storage = createSubBuffer(f, m_bwt_allocation_size * 2);
}
