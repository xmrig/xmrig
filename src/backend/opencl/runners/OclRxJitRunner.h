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

#ifndef XMRIG_OCLRXJITRUNNER_H
#define XMRIG_OCLRXJITRUNNER_H


#include "backend/opencl/runners/OclRxBaseRunner.h"


namespace xmrig {


class RxJitKernel;
class RxRunKernel;


class OclRxJitRunner : public OclRxBaseRunner
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(OclRxJitRunner)

    OclRxJitRunner(size_t index, const OclLaunchData &data);
    ~OclRxJitRunner() override;

protected:
    size_t bufferSize() const override;
    void build() override;
    void execute(uint32_t iteration) override;
    void init() override;

private:
    bool loadAsmProgram();

    cl_mem m_intermediate_programs  = nullptr;
    cl_mem m_programs               = nullptr;
    cl_mem m_registers              = nullptr;
    cl_program m_asmProgram         = nullptr;
    RxJitKernel *m_randomx_jit      = nullptr;
    RxRunKernel *m_randomx_run      = nullptr;
};


} /* namespace xmrig */


#endif // XMRIG_OCLRXRUNNER_H
