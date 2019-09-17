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


#include "backend/common/Tags.h"
#include "backend/opencl/wrappers/OclError.h"
#include "backend/opencl/wrappers/OclKernel.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"


#include <stdexcept>


xmrig::OclKernel::OclKernel(cl_program program, const char *name) :
    m_name(name)
{
    m_kernel = OclLib::createKernel(program, name);
}


xmrig::OclKernel::~OclKernel()
{
    OclLib::release(m_kernel);
}


void xmrig::OclKernel::enqueueNDRange(cl_command_queue queue, uint32_t work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size)
{
    const cl_int ret = OclLib::enqueueNDRangeKernel(queue, m_kernel, work_dim, global_work_offset, global_work_size, local_work_size, 0, nullptr, nullptr);
    if (ret != CL_SUCCESS) {
        LOG_ERR("%s" RED(" error ") RED_BOLD("%s") RED(" when calling ") RED_BOLD("clEnqueueNDRangeKernel") RED(" for kernel ") RED_BOLD("%s"),
                ocl_tag(), OclError::toString(ret), name().data());

        throw std::runtime_error(OclError::toString(ret));
    }
}


void xmrig::OclKernel::setArg(uint32_t index, size_t size, const void *value)
{
    const cl_int ret = OclLib::setKernelArg(m_kernel, index, size, value);
    if (ret != CL_SUCCESS) {
        LOG_ERR("%s" RED(" error ") RED_BOLD("%s") RED(" when calling ") RED_BOLD("clSetKernelArg") RED(" for kernel ") RED_BOLD("%s")
                RED(" argument ") RED_BOLD("%u") RED(" size ") RED_BOLD("%zu"),
                ocl_tag(), OclError::toString(ret), name().data(), index, size);

        throw std::runtime_error(OclError::toString(ret));
    }
}
