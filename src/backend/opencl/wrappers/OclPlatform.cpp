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

#include "backend/opencl/wrappers/OclLib.h"
#include "backend/opencl/wrappers/OclPlatform.h"


std::vector<xmrig::OclPlatform> xmrig::OclPlatform::get()
{
    const std::vector<cl_platform_id> platforms = OclLib::getPlatformIDs();
    std::vector<OclPlatform> out;
    if (platforms.empty()) {
        return out;
    }

    out.reserve(platforms.size());

    for (size_t i = 0; i < platforms.size(); i++) {
        out.emplace_back(i, platforms[i]);
    }

    return out;
}


xmrig::String xmrig::OclPlatform::vendor() const
{
    constexpr size_t size = 128;
    char *buf = new char[size]();

    if (OclLib::getPlatformInfo(id(), CL_PLATFORM_VENDOR, size, buf, nullptr) != CL_SUCCESS) {
        delete [] buf;

        return String();
    }

    return String(buf);
}
