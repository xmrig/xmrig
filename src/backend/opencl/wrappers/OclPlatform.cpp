/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "backend/opencl/wrappers/OclPlatform.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/opencl/wrappers/OclLib.h"


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


void xmrig::OclPlatform::print()
{
    const auto platforms = OclPlatform::get();

    printf("%-28s%zu\n\n", "Number of platforms:", platforms.size());

    for (const auto &platform : platforms) {
        printf("  %-26s%zu\n",  "Index:",       platform.index());
        printf("  %-26s%s\n",   "Profile:",     platform.profile().data());
        printf("  %-26s%s\n",   "Version:",     platform.version().data());
        printf("  %-26s%s\n",   "Name:",        platform.name().data());
        printf("  %-26s%s\n",   "Vendor:",      platform.vendor().data());
        printf("  %-26s%s\n\n", "Extensions:",  platform.extensions().data());
    }
}


rapidjson::Value xmrig::OclPlatform::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    if (!isValid()) {
        return Value(kNullType);
    }

    Value out(kObjectType);
    out.AddMember("index",      static_cast<uint64_t>(index()), allocator);
    out.AddMember("profile",    profile().toJSON(doc), allocator);
    out.AddMember("version",    version().toJSON(doc), allocator);
    out.AddMember("name",       name().toJSON(doc), allocator);
    out.AddMember("vendor",     vendor().toJSON(doc), allocator);
    out.AddMember("extensions", extensions().toJSON(doc), allocator);

    return out;
}


std::vector<xmrig::OclDevice> xmrig::OclPlatform::devices() const
{
    std::vector<OclDevice> out;
    if (!isValid()) {
        return out;
    }

    cl_uint num_devices = 0;
    OclLib::getDeviceIDs(id(), CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (num_devices == 0) {
        return out;
    }

    out.reserve(num_devices);
    std::vector<cl_device_id> devices(num_devices);
    OclLib::getDeviceIDs(id(), CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

    for (size_t i = 0; i < devices.size(); ++i) {
        out.emplace_back(i, devices[i], id());
    }

    return out;
}


xmrig::String xmrig::OclPlatform::extensions() const
{
    return OclLib::getString(id(), CL_PLATFORM_EXTENSIONS);
}


xmrig::String xmrig::OclPlatform::name() const
{
    return OclLib::getString(id(), CL_PLATFORM_NAME);
}


xmrig::String xmrig::OclPlatform::profile() const
{
    return OclLib::getString(id(), CL_PLATFORM_PROFILE);
}


xmrig::String xmrig::OclPlatform::vendor() const
{
    return OclLib::getString(id(), CL_PLATFORM_VENDOR);
}


xmrig::String xmrig::OclPlatform::version() const
{
    return OclLib::getString(id(), CL_PLATFORM_VERSION);
}
