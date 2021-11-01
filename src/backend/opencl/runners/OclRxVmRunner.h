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

#ifndef XMRIG_OCLRXVMRUNNER_H
#define XMRIG_OCLRXVMRUNNER_H


#include "backend/opencl/runners/OclRxBaseRunner.h"


namespace xmrig {


class ExecuteVmKernel;
class InitVmKernel;


class OclRxVmRunner : public OclRxBaseRunner
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(OclRxVmRunner)

    OclRxVmRunner(size_t index, const OclLaunchData &data);
    ~OclRxVmRunner() override;

protected:
    size_t bufferSize() const override;
    void build() override;
    void execute(uint32_t iteration) override;
    void init() override;

private:
    cl_mem m_vm_states              = nullptr;
    ExecuteVmKernel *m_execute_vm   = nullptr;
    InitVmKernel *m_init_vm         = nullptr;
};


} /* namespace xmrig */


#endif // XMRIG_OCLRXVMRUNNER_H
