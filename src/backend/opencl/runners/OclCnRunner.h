/* xmlcore
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_OCLCNRUNNER_H
#define xmlcore_OCLCNRUNNER_H


#include "backend/opencl/runners/OclBaseRunner.h"


namespace xmlcore {


class Cn0Kernel;
class Cn1Kernel;
class Cn2Kernel;
class CnBranchKernel;


class OclCnRunner : public OclBaseRunner
{
public:
    xmlcore_DISABLE_COPY_MOVE_DEFAULT(OclCnRunner)

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


} /* namespace xmlcore */


#endif // xmlcore_OCLCNRUNNER_H
