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

#include "backend/opencl/runners/OclRxVmRunner.h"

#include "backend/opencl/kernels/rx/Blake2bHashRegistersKernel.h"
#include "backend/opencl/kernels/rx/ExecuteVmKernel.h"
#include "backend/opencl/kernels/rx/HashAesKernel.h"
#include "backend/opencl/kernels/rx/InitVmKernel.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "crypto/rx/RxAlgo.h"

#include "base/io/log/Log.h"


xmrig::OclRxVmRunner::OclRxVmRunner(size_t index, const OclLaunchData &data) : OclRxBaseRunner(index, data)
{
}


xmrig::OclRxVmRunner::~OclRxVmRunner()
{
    delete m_init_vm;
    delete m_execute_vm;

    OclLib::release(m_vm_states);
}


void xmrig::OclRxVmRunner::build()
{
    OclRxBaseRunner::build();

    const uint32_t batch_size       = data().thread.intensity();
    const uint32_t hashStrideBytes  = RxAlgo::programSize(m_algorithm) * 8;

    m_hashAes1Rx4->setArgs(m_scratchpads, m_vm_states, hashStrideBytes, batch_size);
    m_blake2b_hash_registers_32->setArgs(m_hashes, m_vm_states, hashStrideBytes);
    m_blake2b_hash_registers_64->setArgs(m_hashes, m_vm_states, hashStrideBytes);

    m_init_vm = new InitVmKernel(m_program);
    m_init_vm->setArgs(m_entropy, m_vm_states, m_rounding);

    m_execute_vm = new ExecuteVmKernel(m_program);
    m_execute_vm->setArgs(m_vm_states, m_rounding, m_scratchpads, data().dataset->get(), batch_size);
}


void xmrig::OclRxVmRunner::execute(uint32_t iteration)
{
    const uint32_t bfactor        = std::min(data().thread.bfactor(), 8u);
    const uint32_t num_iterations = RxAlgo::programIterations(m_algorithm) >> bfactor;
    const uint32_t g_intensity    = data().thread.intensity();

    m_init_vm->enqueue(m_queue, g_intensity, iteration);

//    LOG_WARN("bfactor:%u %u %u", bfactor, RxAlgo::programIterations(m_algorithm), num_iterations);

    uint32_t first = 1;
    uint32_t last  = 0;

    m_execute_vm->setIterations(num_iterations);
    m_execute_vm->setFirst(first);
    m_execute_vm->setLast(last);

    for (int j = 0, n = 1 << bfactor; j < n; ++j) {
        if (j == n - 1) {
            last = 1;
            m_execute_vm->setLast(last);
        }

        m_execute_vm->enqueue(m_queue, g_intensity, data().thread.worksize());

        if (j == 0) {
            first = 0;
            m_execute_vm->setFirst(first);
        }
    }
}


void xmrig::OclRxVmRunner::init()
{
    OclRxBaseRunner::init();

    m_vm_states = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE, 2560 * data().thread.intensity());
}
