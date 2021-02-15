/* xmlcore
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_EXECUTEVMKERNEL_H
#define xmlcore_EXECUTEVMKERNEL_H


#include "backend/opencl/wrappers/OclKernel.h"


namespace xmlcore {


class ExecuteVmKernel : public OclKernel
{
public:
    inline ExecuteVmKernel(cl_program program) : OclKernel(program, "execute_vm") {}

    void enqueue(cl_command_queue queue, size_t threads, size_t worksize);
    void setArgs(cl_mem vm_states, cl_mem rounding, cl_mem scratchpads, cl_mem dataset_ptr, uint32_t batch_size);
    void setFirst(uint32_t first);
    void setIterations(uint32_t num_iterations);
    void setLast(uint32_t last);
};


} // namespace xmlcore


#endif /* xmlcore_EXECUTEVMKERNEL_H */
