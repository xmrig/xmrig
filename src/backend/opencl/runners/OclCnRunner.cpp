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

#include <stdexcept>


#include "backend/opencl/runners/OclCnRunner.h"
#include "backend/opencl/kernels/Cn0Kernel.h"
#include "backend/opencl/kernels/Cn1Kernel.h"
#include "backend/opencl/kernels/Cn2Kernel.h"
#include "backend/opencl/kernels/CnBranchKernel.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/runners/tools/OclCnR.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "crypto/cn/CnAlgo.h"


xmrig::OclCnRunner::OclCnRunner(size_t index, const OclLaunchData &data) : OclBaseRunner(index, data)
{
    uint32_t stridedIndex = data.thread.stridedIndex();
    const auto f = m_algorithm.family();

    if (data.device.vendorId() == OCL_VENDOR_NVIDIA) {
        stridedIndex = 0;
    }
    else if (stridedIndex == 1 && (f == Algorithm::CN_PICO || f == Algorithm::CN_FEMTO || (f == Algorithm::CN && m_algorithm.base() == Algorithm::CN_2))) {
        stridedIndex = 2;
    }

    m_options += " -DITERATIONS="           + std::to_string(CnAlgo<>::iterations(m_algorithm)) + "U";
    m_options += " -DMASK="                 + std::to_string(CnAlgo<>::mask(m_algorithm)) + "U";
    m_options += " -DWORKSIZE="             + std::to_string(data.thread.worksize()) + "U";
    m_options += " -DSTRIDED_INDEX="        + std::to_string(stridedIndex) + "U";
    m_options += " -DMEM_CHUNK_EXPONENT="   + std::to_string(1U << data.thread.memChunk()) + "U";
    m_options += " -DMEMORY="               + std::to_string(m_algorithm.l3()) + "LU";
    m_options += " -DALGO="                 + std::to_string(m_algorithm.id());
    m_options += " -DALGO_BASE="            + std::to_string(m_algorithm.base());
    m_options += " -DALGO_FAMILY="          + std::to_string(m_algorithm.family());
    m_options += " -DCN_UNROLL="            + std::to_string(data.thread.unrollFactor());
}


xmrig::OclCnRunner::~OclCnRunner()
{
    delete m_cn0;
    delete m_cn1;
    delete m_cn2;

    OclLib::release(m_scratchpads);
    OclLib::release(m_states);

    for (size_t i = 0; i < BRANCH_MAX; ++i) {
        delete m_branchKernels[i];
        OclLib::release(m_branches[i]);
    }

    if (m_algorithm == Algorithm::CN_R) {
        OclLib::release(m_cnr);
        OclCnR::clear();
    }
}


size_t xmrig::OclCnRunner::bufferSize() const
{
    return OclBaseRunner::bufferSize() +
           align(m_algorithm.l3() * m_intensity) +
           align(200 * m_intensity) +
           (align(sizeof(cl_uint) * (m_intensity + 2)) * BRANCH_MAX);
}


void xmrig::OclCnRunner::run(uint32_t nonce, uint32_t *hashOutput)
{
    static const cl_uint zero = 0;

    const size_t w_size = data().thread.worksize();
    const size_t g_thd  = ((m_intensity + w_size - 1U) / w_size) * w_size;

    assert(g_thd % w_size == 0);

    for (size_t i = 0; i < BRANCH_MAX; ++i) {
        enqueueWriteBuffer(m_branches[i], CL_FALSE, sizeof(cl_uint) * m_intensity, sizeof(cl_uint), &zero);
    }

    enqueueWriteBuffer(m_output, CL_FALSE, sizeof(cl_uint) * 0xFF, sizeof(cl_uint), &zero);

    m_cn0->enqueue(m_queue, nonce, g_thd);
    m_cn1->enqueue(m_queue, nonce, g_thd, w_size);
    m_cn2->enqueue(m_queue, nonce, g_thd);

    for (auto kernel : m_branchKernels) {
        kernel->enqueue(m_queue, nonce, g_thd, w_size);
    }

    finalize(hashOutput);
}


void xmrig::OclCnRunner::set(const Job &job, uint8_t *blob)
{
    if (job.size() > (Job::kMaxBlobSize - 4)) {
        throw std::length_error("job size too big");
    }

    const int inlen = static_cast<int>(job.size() + 136 - (job.size() % 136));

    blob[job.size()] = 0x01;
    memset(blob + job.size() + 1, 0, inlen - job.size() - 1);

    blob[inlen - 1] |= 0x80;

    enqueueWriteBuffer(m_input, CL_TRUE, 0, inlen, blob);

    m_cn0->setArg(1, sizeof(int), &inlen);

    if (m_algorithm == Algorithm::CN_R && m_height != job.height()) {
        delete m_cn1;

        m_height     = job.height();
        auto program = OclCnR::get(*this, m_height);
        m_cn1        = new Cn1Kernel(program, m_height);
        m_cn1->setArgs(m_input, m_scratchpads, m_states, m_intensity);

        if (m_cnr != program) {
            OclLib::release(m_cnr);
            m_cnr = OclLib::retain(program);
        }
    }

    for (auto kernel : m_branchKernels) {
        kernel->setTarget(job.target());
    }
}


void xmrig::OclCnRunner::build()
{
    OclBaseRunner::build();

    m_cn0 = new Cn0Kernel(m_program);
    m_cn0->setArgs(m_input, 0, m_scratchpads, m_states, m_intensity);

    m_cn2 = new Cn2Kernel(m_program);
    m_cn2->setArgs(m_scratchpads, m_states, m_branches, m_intensity);

    if (m_algorithm != Algorithm::CN_R) {
        m_cn1 = new Cn1Kernel(m_program);
        m_cn1->setArgs(m_input, m_scratchpads, m_states, m_intensity);
    }

    for (size_t i = 0; i < BRANCH_MAX; ++i) {
        auto kernel = new CnBranchKernel(i, m_program);
        kernel->setArgs(m_states, m_branches[i], m_output, m_intensity);

        m_branchKernels[i] = kernel;
    }
}


void xmrig::OclCnRunner::init()
{
    OclBaseRunner::init();

    m_scratchpads = createSubBuffer(CL_MEM_READ_WRITE, m_algorithm.l3() * m_intensity);
    m_states      = createSubBuffer(CL_MEM_READ_WRITE, 200 * m_intensity);

    for (size_t i = 0; i < BRANCH_MAX; ++i) {
        m_branches[i] = createSubBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint) * (m_intensity + 2));
    }
}
