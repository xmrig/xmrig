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

#ifndef XMRIG_OCLCONTEXT_H
#define XMRIG_OCLCONTEXT_H


#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/wrappers/OclDevice.h"
#include "base/tools/Object.h"


using cl_context = struct _cl_context *;


namespace xmrig {


class Job;


class OclContext
{
public:
    XMRIG_DISABLE_COPY_MOVE(OclContext)

    OclContext() = default;
    OclContext(const OclDevice &device);
    ~OclContext();

    bool init(const std::vector<OclDevice> &devices, std::vector<OclLaunchData> &threads);

    inline bool isValid() const     { return m_ctx != nullptr; }
    inline cl_context ctx() const   { return m_ctx; }

private:
    cl_context m_ctx = nullptr;
};


} // namespace xmrig


#endif /* XMRIG_OCLCONTEXT_H */
