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

#include "backend/opencl/runners/OclRxJitRunner.h"

#include "backend/opencl/wrappers/OclLib.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/kernels/rx/HashAesKernel.h"
#include "backend/opencl/kernels/rx/Blake2bHashRegistersKernel.h"


xmrig::OclRxJitRunner::OclRxJitRunner(size_t index, const OclLaunchData &data) : OclRxBaseRunner(index, data)
{
}


xmrig::OclRxJitRunner::~OclRxJitRunner()
{
    OclLib::release(m_intermediate_programs);
    OclLib::release(m_programs);
    OclLib::release(m_registers);
}


void xmrig::OclRxJitRunner::build()
{
    OclRxBaseRunner::build();

    const uint32_t batch_size = data().thread.intensity();

    m_hashAes1Rx4->setArgs(m_scratchpads, m_registers, 256, batch_size);
    m_blake2b_hash_registers_32->setArgs(m_hashes, m_registers, 256);
    m_blake2b_hash_registers_64->setArgs(m_hashes, m_registers, 256);
}


void xmrig::OclRxJitRunner::execute(uint32_t iteration)
{
}


void xmrig::OclRxJitRunner::init()
{
    OclRxBaseRunner::init();

    const size_t g_thd = data().thread.intensity();

    m_registers             = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE, 256 * g_thd, nullptr);
    m_intermediate_programs = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE, 5120 * g_thd, nullptr);
    m_programs              = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE, 10048 * g_thd, nullptr);
}
