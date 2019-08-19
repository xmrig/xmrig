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


#include "backend/opencl/OclConfig.h"
#include "base/io/json/Json.h"
#include "rapidjson/document.h"


namespace xmrig {

static const char *kCache       = "cache";
static const char *kEnabled     = "enabled";
static const char *kLoader      = "loader";
static const char *kPlatform    = "platform";
static const char *kAMD         = "AMD";
static const char *kNVIDIA      = "NVIDIA";
static const char *kINTEL       = "INTEL";


extern template class Threads<OclThreads>;

}


xmrig::OclConfig::OclConfig() :
    m_platformVendor(kAMD)
{
}


xmrig::OclPlatform xmrig::OclConfig::platform() const
{
    const auto platforms = OclPlatform::get();
    if (platforms.empty()) {
        return OclPlatform();
    }

    if (!m_platformVendor.isEmpty()) {
        String search;
        String vendor = m_platformVendor;
        vendor.toUpper();

        if (vendor == kAMD) {
            search = "Advanced Micro Devices";
        }
        else if (vendor == kNVIDIA) {
            search = kNVIDIA;
        }
        else if (vendor == kINTEL) {
            search = "Intel";
        }
        else {
            search = m_platformVendor;
        }

        for (const auto &platform : platforms) {
            if (platform.vendor().contains(search)) {
                return platform;
            }
        }
    }
    else if (m_platformIndex < platforms.size()) {
        return platforms[m_platformIndex];
    }

    return OclPlatform();
}


rapidjson::Value xmrig::OclConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);

    obj.AddMember(StringRef(kEnabled),  m_enabled, allocator);
    obj.AddMember(StringRef(kCache),    m_cache, allocator);
    obj.AddMember(StringRef(kLoader),   m_loader.toJSON(), allocator);
    obj.AddMember(StringRef(kPlatform), m_platformVendor.isEmpty() ? Value(m_platformIndex) : m_platformVendor.toJSON(), allocator);

    m_threads.toJSON(obj, doc);

    return obj;
}


std::vector<xmrig::OclLaunchData> xmrig::OclConfig::get(const Miner *miner, const Algorithm &algorithm) const
{
    std::vector<OclLaunchData> out;

    return out;
}


void xmrig::OclConfig::read(const rapidjson::Value &value)
{
    if (value.IsObject()) {
        m_enabled   = Json::getBool(value, kEnabled, m_enabled);
        m_cache     = Json::getBool(value, kCache, m_cache);
        m_loader    = Json::getString(value, kLoader);

        setPlatform(Json::getValue(value, kPlatform));

        if (!m_threads.read(value)) {
            generate();
        }
    }
    else if (value.IsBool() && value.IsFalse()) {
        m_enabled = false;
    }
    else {
        generate();
    }
}


void xmrig::OclConfig::generate()
{
    m_shouldSave  = true;
}


void xmrig::OclConfig::setPlatform(const rapidjson::Value &platform)
{
    if (platform.IsString()) {
        m_platformVendor = platform.GetString();
    }
    else if (platform.IsUint()) {
        m_platformVendor = nullptr;
        m_platformIndex  = platform.GetUint();
    }
}
