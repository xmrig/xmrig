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


#include "backend/opencl/kernels/rx/HashAesKernel.h"
#include "backend/opencl/wrappers/OclLib.h"


void xmrig::HashAesKernel::enqueue(cl_command_queue queue, size_t threads)
{
    const size_t gthreads        = threads * 4;
    static const size_t lthreads = 64;

    enqueueNDRange(queue, 1, nullptr, &gthreads, &lthreads);
}


// __kernel void hashAes1Rx4(__global const void* input, __global void* hash, uint hashOffsetBytes, uint hashStrideBytes, uint batch_size)
void xmrig::HashAesKernel::setArgs(cl_mem input, cl_mem hash, uint32_t hashStrideBytes, uint32_t batch_size)
{
    const uint32_t hashOffsetBytes = 192;

    setArg(0, sizeof(cl_mem), &input);
    setArg(1, sizeof(cl_mem), &hash);
    setArg(2, sizeof(uint32_t), &hashOffsetBytes);
    setArg(3, sizeof(uint32_t), &hashStrideBytes);
    setArg(4, sizeof(uint32_t), &batch_size);
}
