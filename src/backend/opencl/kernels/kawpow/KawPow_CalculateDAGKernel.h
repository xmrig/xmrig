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

#ifndef XMRIG_KAWPOW_CALCULATEDAGKERNEL_H
#define XMRIG_KAWPOW_CALCULATEDAGKERNEL_H


#include "backend/opencl/wrappers/OclKernel.h"


namespace xmrig {


class KawPow_CalculateDAGKernel : public OclKernel
{
public:
    inline KawPow_CalculateDAGKernel(cl_program program) : OclKernel(program, "ethash_calculate_dag_item") {}

    void enqueue(cl_command_queue queue, size_t threads, size_t workgroup_size);
    void setArgs(uint32_t start, cl_mem g_light, cl_mem g_dag, uint32_t dag_words, uint32_t light_words);
};


} // namespace xmrig


#endif /* XMRIG_KAWPOW_CALCULATEDAGKERNEL_H */
