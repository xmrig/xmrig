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

#ifndef XMRIG_OCLCNRUNNER_H
#define XMRIG_OCLCNRUNNER_H


#include "backend/opencl/runners/OclBaseRunner.h"


namespace xmrig {


class Cn0Kernel;
class Cn1Kernel;
class Cn2Kernel;
class CnBranchKernel;


class OclCnRunner : public OclBaseRunner
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(OclCnRunner)

    OclCnRunner(size_t index, const OclLaunchData &data);
    ~OclCnRunner() override;

protected:
    size_t bufferSize() const override;
    void run(uint32_t nonce, uint32_t *hashOutput) override;
    void set(const Job &job, uint8_t *blob) override;
    void build() override;
    void init() override;

private:
    enum Branches : size_t {
        BRANCH_BLAKE_256,
        BRANCH_GROESTL_256,
        BRANCH_JH_256,
        BRANCH_SKEIN_512,
        BRANCH_MAX
    };


    cl_mem m_scratchpads    = nullptr;
    cl_mem m_states         = nullptr;
    cl_program m_cnr        = nullptr;
    Cn0Kernel *m_cn0        = nullptr;
    Cn1Kernel *m_cn1        = nullptr;
    Cn2Kernel *m_cn2        = nullptr;
    uint64_t m_height       = 0;

    std::vector<cl_mem> m_branches                = { nullptr, nullptr, nullptr, nullptr };
    std::vector<CnBranchKernel *> m_branchKernels = { nullptr, nullptr, nullptr, nullptr };
};


} /* namespace xmrig */


#endif // XMRIG_OCLCNRUNNER_H
