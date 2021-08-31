/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "KawPow_CalculateDAGKernel.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "crypto/kawpow/KPCache.h"


void xmrig::KawPow_CalculateDAGKernel::enqueue(cl_command_queue queue, size_t threads, size_t workgroup_size)
{
    enqueueNDRange(queue, 1, nullptr, &threads, &workgroup_size);
}


void xmrig::KawPow_CalculateDAGKernel::setArgs(uint32_t start, cl_mem g_light, cl_mem g_dag, uint32_t dag_words, uint32_t light_words)
{
    setArg(0, sizeof(start), &start);
    setArg(1, sizeof(cl_mem), &g_light);
    setArg(2, sizeof(cl_mem), &g_dag);

    const uint32_t isolate = 1;
    setArg(3, sizeof(isolate), &isolate);

    setArg(4, sizeof(dag_words), &dag_words);

    uint32_t light_words4[4];
    KPCache::calculate_fast_mod_data(light_words, light_words4[0], light_words4[1], light_words4[2]);
    light_words4[3] = light_words;

    setArg(5, sizeof(light_words4), light_words4);
}
