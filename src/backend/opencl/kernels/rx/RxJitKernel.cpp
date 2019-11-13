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


#include "backend/opencl/kernels/rx/RxJitKernel.h"
#include "backend/opencl/wrappers/OclLib.h"


void xmrig::RxJitKernel::enqueue(cl_command_queue queue, size_t threads, uint32_t iteration)
{
    setArg(6, sizeof(uint32_t), &iteration);

    const size_t gthreads        = threads * 32;
    static const size_t lthreads = 64;

    enqueueNDRange(queue, 1, nullptr, &gthreads, &lthreads);
}


// __kernel void randomx_jit(__global ulong* entropy, __global ulong* registers, __global uint2* intermediate_programs, __global uint* programs, uint batch_size, __global uint32_t* rounding, uint32_t iteration)
void xmrig::RxJitKernel::setArgs(cl_mem entropy, cl_mem registers, cl_mem intermediate_programs, cl_mem programs, uint32_t batch_size, cl_mem rounding)
{
    setArg(0, sizeof(cl_mem), &entropy);
    setArg(1, sizeof(cl_mem), &registers);
    setArg(2, sizeof(cl_mem), &intermediate_programs);
    setArg(3, sizeof(cl_mem), &programs);
    setArg(4, sizeof(uint32_t), &batch_size);
    setArg(5, sizeof(cl_mem), &rounding);
}
