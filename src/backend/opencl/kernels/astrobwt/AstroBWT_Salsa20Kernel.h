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

#ifndef XMRIG_ASTROBWT_SALSA20KERNEL_H
#define XMRIG_ASTROBWT_SALSA20KERNEL_H


#include "backend/opencl/wrappers/OclKernel.h"


namespace xmrig {


class AstroBWT_Salsa20Kernel : public OclKernel
{
public:
    inline AstroBWT_Salsa20Kernel(cl_program program) : OclKernel(program, "Salsa20_XORKeyStream") {}

    void enqueue(cl_command_queue queue, size_t threads, size_t workgroup_size);
    void setArgs(cl_mem salsa20_keys, cl_mem outputs, cl_mem output_sizes, uint32_t output_stride);
};


} // namespace xmrig


#endif /* XMRIG_ASTROBWT_SALSA20KERNEL_H */
