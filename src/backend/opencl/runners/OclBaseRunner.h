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

#ifndef XMRIG_OCLBASERUNNER_H
#define XMRIG_OCLBASERUNNER_H


#include <string>


#include "3rdparty/cl.h"
#include "backend/opencl/interfaces/IOclRunner.h"
#include "crypto/common/Algorithm.h"


namespace xmrig {


class OclLaunchData;


class OclBaseRunner : public IOclRunner
{
public:
    OclBaseRunner(size_t index, const OclLaunchData &data);
    ~OclBaseRunner() override;

protected:
    bool selfTest() const override;
    const char *buildOptions() const override;
    void run(uint32_t *hashOutput) override;
    void set(const Job &job) override;

protected:
    Algorithm m_algorithm;
    cl_command_queue m_queue    = nullptr;
    cl_context m_ctx;
    cl_mem m_input              = nullptr;
    cl_mem m_output             = nullptr;
    std::string m_options;
};


} /* namespace xmrig */


#endif // XMRIG_OCLBASERUNNER_H
