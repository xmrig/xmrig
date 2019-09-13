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


#include <string>


#include "backend/opencl/kernels/Cn1Kernel.h"
#include "backend/opencl/wrappers/OclLib.h"


xmrig::Cn1Kernel::Cn1Kernel(cl_program program)
    : OclKernel(program, "cn1")
{
}


xmrig::Cn1Kernel::Cn1Kernel(cl_program program, uint64_t height)
    : OclKernel(program, ("cn1_" + std::to_string(height)).c_str())
{

}


void xmrig::Cn1Kernel::enqueue(cl_command_queue queue, uint32_t nonce, size_t threads, size_t worksize)
{
    const size_t offset   = nonce;
    const size_t gthreads = threads;
    const size_t lthreads = worksize;

    enqueueNDRange(queue, 1, &offset, &gthreads, &lthreads);
}


// __kernel void cn1(__global ulong *input, __global uint4 *Scratchpad, __global ulong *states, uint Threads)
void xmrig::Cn1Kernel::setArgs(cl_mem input, cl_mem scratchpads, cl_mem states, uint32_t threads)
{
    setArg(0, sizeof(cl_mem), &input);
    setArg(1, sizeof(cl_mem), &scratchpads);
    setArg(2, sizeof(cl_mem), &states);
    setArg(3, sizeof(uint32_t), &threads);
}
