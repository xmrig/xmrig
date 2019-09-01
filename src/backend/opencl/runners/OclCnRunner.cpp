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


#include "backend/opencl/kernels/Cn0Kernel.h"
#include "backend/opencl/kernels/Cn1Kernel.h"
#include "backend/opencl/kernels/Cn2Kernel.h"
#include "backend/opencl/kernels/CnBranchKernel.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/runners/OclCnRunner.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"
#include "crypto/cn/CnAlgo.h"


xmrig::OclCnRunner::OclCnRunner(size_t index, const OclLaunchData &data) : OclBaseRunner(index, data)
{
    if (m_queue == nullptr) {
        return;
    }

    const size_t g_thd = data.thread.intensity();

    cl_int ret;
    m_scratchpads = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE, data.algorithm.l3() * g_thd, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        return;
    }

    m_states = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE, 200 * g_thd, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        return;
    }

    for (size_t i = 0; i < BRANCH_MAX; ++i) {
        m_branches[i] = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE, sizeof(cl_uint) * (g_thd + 2), nullptr, &ret);
        if (ret != CL_SUCCESS) {
            return;
        }
    }

    uint32_t stridedIndex = data.thread.stridedIndex();
    if (data.device.vendorId() == OCL_VENDOR_NVIDIA) {
        stridedIndex = 0;
    }
    else if (stridedIndex == 1 && (m_algorithm.family() == Algorithm::CN_PICO || (m_algorithm.family() == Algorithm::CN && CnAlgo<>::base(m_algorithm) == Algorithm::CN_2))) {
        stridedIndex = 2;
    }

    m_options += " -DITERATIONS="           + std::to_string(CnAlgo<>::iterations(m_algorithm)) + "U";
    m_options += " -DMASK="                 + std::to_string(CnAlgo<>::mask(m_algorithm)) + "U";
    m_options += " -DWORKSIZE="             + std::to_string(data.thread.worksize()) + "U";
    m_options += " -DSTRIDED_INDEX="        + std::to_string(stridedIndex) + "U";
    m_options += " -DMEM_CHUNK_EXPONENT="   + std::to_string(1u << data.thread.memChunk()) + "U";
    m_options += " -DCOMP_MODE="            + std::to_string(data.thread.isCompMode() && g_thd % data.thread.worksize() != 0 ? 1u : 0u) + "U";
    m_options += " -DMEMORY="               + std::to_string(m_algorithm.l3()) + "LU";
    m_options += " -DALGO="                 + std::to_string(m_algorithm.id());
    m_options += " -DALGO_FAMILY="          + std::to_string(m_algorithm.family());
    m_options += " -DCN_UNROLL="            + std::to_string(data.thread.unrollFactor());

#   ifdef XMRIG_ALGO_CN_GPU
    if (data.algorithm == Algorithm::CN_GPU) {
        m_options += " -cl-fp32-correctly-rounded-divide-sqrt";
    }
#   endif
}


xmrig::OclCnRunner::~OclCnRunner()
{
    delete m_cn0;
    delete m_cn1;
    delete m_cn2;

    OclLib::releaseMemObject(m_scratchpads);
    OclLib::releaseMemObject(m_states);

    for (size_t i = 0; i < BRANCH_MAX; ++i) {
        delete m_branchKernels[i];
        OclLib::releaseMemObject(m_branches[i]);
    }
}


bool xmrig::OclCnRunner::isReadyToBuild() const
{
    return OclBaseRunner::isReadyToBuild() &&
            m_scratchpads                   != nullptr &&
            m_states                        != nullptr &&
            m_branches[BRANCH_BLAKE_256]    != nullptr &&
            m_branches[BRANCH_GROESTL_256]  != nullptr &&
            m_branches[BRANCH_JH_256]       != nullptr &&
            m_branches[BRANCH_SKEIN_512]    != nullptr;
}


bool xmrig::OclCnRunner::run(uint32_t nonce, uint32_t *hashOutput)
{
    static const cl_uint zero = 0;

    const size_t g_intensity = data().thread.intensity();
    const size_t w_size      = data().thread.worksize();
    const size_t g_thd       = ((g_intensity + w_size - 1u) / w_size) * w_size;

    assert(g_thd % w_size == 0);

    for (size_t i = 0; i < BRANCH_MAX; ++i) {
        if (OclLib::enqueueWriteBuffer(m_queue, m_branches[i], CL_FALSE, sizeof(cl_uint) * g_intensity, sizeof(cl_uint), &zero, 0, nullptr, nullptr) != CL_SUCCESS) {
            return false;
        }
    }

    if (OclLib::enqueueWriteBuffer(m_queue, m_output, CL_FALSE, sizeof(cl_uint) * 0xFF, sizeof(cl_uint), &zero, 0, nullptr, nullptr) != CL_SUCCESS) {
        return false;
    }

    if (!m_cn0->enqueue(m_queue, nonce, g_thd)) {
        return false;
    }

    if (!m_cn1->enqueue(m_queue, nonce, g_thd, w_size)) {
        return false;
    }

    if (!m_cn2->enqueue(m_queue, nonce, g_thd)) {
        return false;
    }

    for (size_t i = 0; i < BRANCH_MAX; ++i) {
        if (!m_branchKernels[i]->enqueue(m_queue, nonce, g_thd, w_size)) {
            return false;
        }
    }

    if (OclLib::enqueueReadBuffer(m_queue, m_output, CL_TRUE, 0, sizeof(cl_uint) * 0x100, hashOutput, 0, nullptr, nullptr) != CL_SUCCESS) {
        return false;
    }

    uint32_t &results = hashOutput[0xFF];
    if (results > 0xFF) {
        results = 0xFF;
    }

    return true;
}


bool xmrig::OclCnRunner::selfTest() const
{
    return OclBaseRunner::selfTest() &&
           m_cn0->isValid() &&
           m_cn1->isValid() && m_cn2->isValid();
}


bool xmrig::OclCnRunner::set(const Job &job, uint8_t *blob)
{
    if (job.size() > (Job::kMaxBlobSize - 4)) {
        return false;
    }

    blob[job.size()] = 0x01;
    memset(blob + job.size() + 1, 0, Job::kMaxBlobSize - job.size() - 1);

    if (OclLib::enqueueWriteBuffer(m_queue, m_input, CL_TRUE, 0, Job::kMaxBlobSize, blob, 0, nullptr, nullptr) != CL_SUCCESS) {
        return false;
    }

    if (!m_cn0->setArgs(m_input, m_scratchpads, m_states)) {
        return false;
    }

    if (!m_cn1->setArgs(m_input, m_scratchpads, m_states)) {
        return false;
    }

    const uint32_t intensity = data().thread.intensity();

    if (!m_cn2->setArgs(m_scratchpads, m_states, m_branches, intensity)) {
        return false;
    }

    for (size_t i = 0; i < BRANCH_MAX; ++i) {
        if (!m_branchKernels[i]->setArgs(m_states, m_branches[i], m_output, job.target(), intensity)) {
            return false;
        }
    }

    return true;
}


void xmrig::OclCnRunner::build()
{
    OclBaseRunner::build();

    if (!m_program) {
        return;
    }

    m_cn0 = new Cn0Kernel(m_program);
    m_cn1 = new Cn1Kernel(m_program);
    m_cn2 = new Cn2Kernel(m_program);

    for (size_t i = 0; i < BRANCH_MAX; ++i) {
        m_branchKernels[i] = new CnBranchKernel(i, m_program);
    }
}
