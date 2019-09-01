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

#ifndef XMRIG_OCLCNRUNNER_H
#define XMRIG_OCLCNRUNNER_H


#include "backend/opencl/runners/OclBaseRunner.h"


namespace xmrig {


class Cn0Kernel;


class OclCnRunner : public OclBaseRunner
{
public:
    OclCnRunner(size_t index, const OclLaunchData &data);
    ~OclCnRunner() override;

protected:
    bool isReadyToBuild() const override;
    bool run(uint32_t nonce, uint32_t *hashOutput) override;
    bool selfTest() const override;
    bool set(const Job &job, uint8_t *blob) override;
    void build() override;

private:
    enum Branches : size_t {
        BRANCH_BLAKE_256,
        BRANCH_GROESTL_256,
        BRANCH_JH_256,
        BRANCH_SKEIN_512,
        BRANCH_MAX
    };


    cl_mem m_branches[BRANCH_MAX]   = { nullptr, nullptr, nullptr, nullptr };
    cl_mem m_scratchpads            = nullptr;
    cl_mem m_states                 = nullptr;
    Cn0Kernel *m_cn0                = nullptr;
};


} /* namespace xmrig */


#endif // XMRIG_OCLCNRUNNER_H
