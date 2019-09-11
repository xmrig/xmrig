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


#include "backend/opencl/kernels/rx/ExecuteVmKernel.h"
#include "backend/opencl/wrappers/OclLib.h"


void xmrig::ExecuteVmKernel::enqueue(cl_command_queue queue, size_t threads, size_t worksize)
{
    const size_t gthreads = (worksize == 16) ? (threads * 16) : (threads * 8);
    const size_t lthreads = (worksize == 16) ? 32 : 16;

    enqueueNDRange(queue, 1, nullptr, &gthreads, &lthreads);
}


// __kernel void execute_vm(__global void* vm_states, __global void* rounding, __global void* scratchpads, __global const void* dataset_ptr, uint32_t batch_size, uint32_t num_iterations, uint32_t first, uint32_t last)
void xmrig::ExecuteVmKernel::setArgs(cl_mem vm_states, cl_mem rounding, cl_mem scratchpads, cl_mem dataset_ptr, uint32_t batch_size)
{
    setArg(0, sizeof(cl_mem), &vm_states);
    setArg(1, sizeof(cl_mem), &rounding);
    setArg(2, sizeof(cl_mem), &scratchpads);
    setArg(3, sizeof(cl_mem), &dataset_ptr);
    setArg(4, sizeof(uint32_t), &batch_size);
}


void xmrig::ExecuteVmKernel::setFirst(uint32_t first)
{
    setArg(6, sizeof(uint32_t), &first);
}


void xmrig::ExecuteVmKernel::setIterations(uint32_t num_iterations)
{
    setArg(5, sizeof(uint32_t), &num_iterations);
    setFirst(1);
    setLast(0);
}


void xmrig::ExecuteVmKernel::setLast(uint32_t last)
{
    setArg(7, sizeof(uint32_t), &last);
}
