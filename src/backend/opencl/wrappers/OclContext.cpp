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


#include "backend/opencl/wrappers/OclContext.h"
#include "backend/opencl/runners/tools/OclSharedState.h"
#include "backend/opencl/wrappers/OclLib.h"


xmrig::OclContext::OclContext(const OclDevice &device)
{
    std::vector<cl_device_id> ids = { device.id() };
    m_ctx = OclLib::createContext(ids);
}


xmrig::OclContext::~OclContext()
{
    if (m_ctx) {
        OclLib::release(m_ctx);
    }
}


bool xmrig::OclContext::init(const std::vector<OclDevice> &devices, std::vector<OclLaunchData> &threads)
{
    if (!m_ctx) {
        std::vector<cl_device_id> ids(devices.size());
        for (size_t i = 0; i < devices.size(); ++i) {
            ids[i] = devices[i].id();
        }

        m_ctx = OclLib::createContext(ids);
    }

    if (!m_ctx) {
        return false;
    }

    for (OclLaunchData &data : threads) {
        data.ctx = m_ctx;
    }

    return true;
}
