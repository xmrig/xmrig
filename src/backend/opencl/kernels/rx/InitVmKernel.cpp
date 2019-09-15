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


#include "backend/opencl/kernels/rx/InitVmKernel.h"
#include "backend/opencl/wrappers/OclLib.h"


void xmrig::InitVmKernel::enqueue(cl_command_queue queue, size_t threads, uint32_t iteration)
{
    setArg(3, sizeof(uint32_t), &iteration);

    const size_t gthreads        = threads * 8;
    static const size_t lthreads = 32;

    enqueueNDRange(queue, 1, nullptr, &gthreads, &lthreads);
}


// __kernel void init_vm(__global const void* entropy_data, __global void* vm_states, __global uint32_t* rounding, uint32_t iteration)
void xmrig::InitVmKernel::setArgs(cl_mem entropy_data, cl_mem vm_states, cl_mem rounding)
{
    setArg(0, sizeof(cl_mem), &entropy_data);
    setArg(1, sizeof(cl_mem), &vm_states);
    setArg(2, sizeof(cl_mem), &rounding);
}
