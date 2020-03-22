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


#include "backend/opencl/kernels/rx/RxRunKernel.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/crypto/Algorithm.h"
#include "crypto/randomx/randomx.h"
#include "crypto/rx/RxAlgo.h"


void xmrig::RxRunKernel::enqueue(cl_command_queue queue, size_t threads, size_t workgroup_size)
{
    const size_t gthreads        = threads * workgroup_size;
    enqueueNDRange(queue, 1, nullptr, &gthreads, &workgroup_size);
}


void xmrig::RxRunKernel::setArgs(cl_mem dataset, cl_mem scratchpads, cl_mem registers, cl_mem rounding, cl_mem programs, uint32_t batch_size, const Algorithm &algorithm)
{
    setArg(0, sizeof(cl_mem), &dataset);
    setArg(1, sizeof(cl_mem), &scratchpads);
    setArg(2, sizeof(cl_mem), &registers);
    setArg(3, sizeof(cl_mem), &rounding);
    setArg(4, sizeof(cl_mem), &programs);
    setArg(5, sizeof(uint32_t), &batch_size);

    auto PowerOf2 = [](size_t N)
    {
        uint32_t result = 0;
        while (N > 1) {
            ++result;
            N >>= 1;
        }

        return result;
    };

    const auto *rx_conf = RxAlgo::base(algorithm);
    const uint32_t rx_parameters =
                    (PowerOf2(rx_conf->ScratchpadL1_Size) << 0) |
                    (PowerOf2(rx_conf->ScratchpadL2_Size) << 5) |
                    (PowerOf2(rx_conf->ScratchpadL3_Size) << 10) |
                    (PowerOf2(rx_conf->ProgramIterations) << 15);

    setArg(6, sizeof(uint32_t), &rx_parameters);
}
