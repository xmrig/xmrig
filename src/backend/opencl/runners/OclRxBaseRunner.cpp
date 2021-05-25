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

#include "backend/opencl/runners/OclRxBaseRunner.h"
#include "backend/opencl/kernels/rx/Blake2bHashRegistersKernel.h"
#include "backend/opencl/kernels/rx/Blake2bInitialHashKernel.h"
#include "backend/opencl/kernels/rx/FillAesKernel.h"
#include "backend/opencl/kernels/rx/FindSharesKernel.h"
#include "backend/opencl/kernels/rx/HashAesKernel.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/runners/tools/OclSharedState.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/net/stratum/Job.h"
#include "crypto/rx/Rx.h"
#include "crypto/rx/RxAlgo.h"
#include "crypto/rx/RxDataset.h"


xmrig::OclRxBaseRunner::OclRxBaseRunner(size_t index, const OclLaunchData &data) : OclBaseRunner(index, data)
{
    switch (data.thread.worksize()) {
    case 2:
    case 4:
    case 8:
    case 16:
        m_worksize = data.thread.worksize();
        break;

    default:
        m_worksize = 8;
    }

    if (data.device.type() == OclDevice::Vega_10 || data.device.type() == OclDevice::Vega_20 || data.device.type() == OclDevice::Raven) {
        m_gcn_version = 14;
    }

    if (data.device.type() == OclDevice::Navi_10 || data.device.type() == OclDevice::Navi_12 || data.device.type() == OclDevice::Navi_14 || data.device.type() == OclDevice::Navi_21) {
        m_gcn_version = 15;
    }

    m_options += " -DALGO="             + std::to_string(RxAlgo::id(m_algorithm));
    m_options += " -DWORKERS_PER_HASH=" + std::to_string(m_worksize);
    m_options += " -DGCN_VERSION="      + std::to_string(m_gcn_version);
}


xmrig::OclRxBaseRunner::~OclRxBaseRunner()
{
    delete m_fillAes1Rx4_scratchpad;
    delete m_fillAes4Rx4_entropy;
    delete m_hashAes1Rx4;
    delete m_blake2b_initial_hash;
    delete m_blake2b_hash_registers_32;
    delete m_blake2b_hash_registers_64;
    delete m_find_shares;

    OclLib::release(m_entropy);
    OclLib::release(m_hashes);
    OclLib::release(m_rounding);
    OclLib::release(m_scratchpads);
    OclLib::release(m_dataset);
}


void xmrig::OclRxBaseRunner::run(uint32_t nonce, uint32_t *hashOutput)
{
    static const uint32_t zero = 0;

    m_blake2b_initial_hash->setNonce(nonce);
    m_find_shares->setNonce(nonce);

    enqueueWriteBuffer(m_output, CL_FALSE, sizeof(cl_uint) * 0xFF, sizeof(uint32_t), &zero);

    m_blake2b_initial_hash->enqueue(m_queue, m_intensity);
    m_fillAes1Rx4_scratchpad->enqueue(m_queue, m_intensity);

    const uint32_t programCount = RxAlgo::programCount(m_algorithm);

    for (uint32_t i = 0; i < programCount; ++i) {
        m_fillAes4Rx4_entropy->enqueue(m_queue, m_intensity);

        execute(i);

        if (i == programCount - 1) {
            m_hashAes1Rx4->enqueue(m_queue, m_intensity);
            m_blake2b_hash_registers_32->enqueue(m_queue, m_intensity);
        }
        else {
            m_blake2b_hash_registers_64->enqueue(m_queue, m_intensity);
        }
    }

    m_find_shares->enqueue(m_queue, m_intensity);

    finalize(hashOutput);

    OclLib::finish(m_queue);
}


void xmrig::OclRxBaseRunner::set(const Job &job, uint8_t *blob)
{
    if (!data().thread.isDatasetHost() && m_seed != job.seed()) {
        m_seed = job.seed();

        auto dataset = Rx::dataset(job, 0);
        enqueueWriteBuffer(m_dataset, CL_TRUE, 0, RxDataset::maxSize(), dataset->raw());
    }

    if (job.size() < Job::kMaxBlobSize) {
        memset(blob + job.size(), 0, Job::kMaxBlobSize - job.size());
    }

    enqueueWriteBuffer(m_input, CL_TRUE, 0, Job::kMaxBlobSize, blob);

    m_blake2b_initial_hash->setBlobSize(job.size());
    m_find_shares->setTarget(job.target());
}


size_t xmrig::OclRxBaseRunner::bufferSize() const
{
    return OclBaseRunner::bufferSize() +
           align((m_algorithm.l3() + 64) * m_intensity) +
           align(64 * m_intensity) +
           align((128 + 2560) * m_intensity) +
           align(sizeof(uint32_t) * m_intensity);
}


void xmrig::OclRxBaseRunner::build()
{
    OclBaseRunner::build();

    const uint32_t rx_version = RxAlgo::version(m_algorithm);

    m_fillAes1Rx4_scratchpad = new FillAesKernel(m_program, "fillAes1Rx4_scratchpad");
    m_fillAes1Rx4_scratchpad->setArgs(m_hashes, m_scratchpads, m_intensity, rx_version);

    m_fillAes4Rx4_entropy = new FillAesKernel(m_program, "fillAes4Rx4_entropy");
    m_fillAes4Rx4_entropy->setArgs(m_hashes, m_entropy, m_intensity, rx_version);

    m_hashAes1Rx4 = new HashAesKernel(m_program);

    m_blake2b_initial_hash = new Blake2bInitialHashKernel(m_program);
    m_blake2b_initial_hash->setArgs(m_hashes, m_input);

    m_blake2b_hash_registers_32 = new Blake2bHashRegistersKernel(m_program, "blake2b_hash_registers_32");
    m_blake2b_hash_registers_64 = new Blake2bHashRegistersKernel(m_program, "blake2b_hash_registers_64");

    m_find_shares = new FindSharesKernel(m_program);
    m_find_shares->setArgs(m_hashes, m_output);
}


void xmrig::OclRxBaseRunner::init()
{
    OclBaseRunner::init();

    m_scratchpads = createSubBuffer(CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, (m_algorithm.l3() + 64) * m_intensity);
    m_hashes      = createSubBuffer(CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, 64 * m_intensity);
    m_entropy     = createSubBuffer(CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, (128 + 2560) * m_intensity);
    m_rounding    = createSubBuffer(CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(uint32_t) * m_intensity);
    m_dataset     = OclSharedState::get(data().device.index()).dataset();
}
