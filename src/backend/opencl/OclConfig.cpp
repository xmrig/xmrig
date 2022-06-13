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

#include "backend/opencl/OclConfig.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/common/Tags.h"
#include "backend/opencl/OclConfig_gen.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/json/Json.h"
#include "base/io/log/Log.h"


namespace xmrig {


static const char *kCache       = "cache";
static const char *kDevicesHint = "devices-hint";
static const char *kEnabled     = "enabled";
static const char *kLoader      = "loader";

#ifndef XMRIG_OS_APPLE
static const char *kAMD         = "AMD";
static const char *kINTEL       = "INTEL";
static const char *kNVIDIA      = "NVIDIA";
static const char *kPlatform    = "platform";
#endif

#ifdef XMRIG_FEATURE_ADL
static const char *kAdl         = "adl";
#endif


extern template class Threads<OclThreads>;


} // namespace xmrig


#ifndef XMRIG_OS_APPLE
xmrig::OclConfig::OclConfig() : m_platformVendor(kAMD) {}
#else
xmrig::OclConfig::OclConfig() = default;
#endif


xmrig::OclPlatform xmrig::OclConfig::platform() const
{
    const auto platforms = OclPlatform::get();
    if (platforms.empty()) {
        return {};
    }

#   ifndef XMRIG_OS_APPLE
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

    return {};
#   else
    return platforms[0];
#   endif
}


rapidjson::Value xmrig::OclConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);

    obj.AddMember(StringRef(kEnabled),  m_enabled, allocator);
    obj.AddMember(StringRef(kCache),    m_cache, allocator);
    obj.AddMember(StringRef(kLoader),   m_loader.toJSON(), allocator);

#   ifndef XMRIG_OS_APPLE
    obj.AddMember(StringRef(kPlatform), m_platformVendor.isEmpty() ? Value(m_platformIndex) : m_platformVendor.toJSON(), allocator);
#   endif

#   ifdef XMRIG_FEATURE_ADL
    obj.AddMember(StringRef(kAdl),      m_adl, allocator);
#   endif

    m_threads.toJSON(obj, doc);

    return obj;
}


std::vector<xmrig::OclLaunchData> xmrig::OclConfig::get(const Miner *miner, const Algorithm &algorithm, const OclPlatform &platform, const std::vector<OclDevice> &devices) const
{
    std::vector<OclLaunchData> out;
    const auto &threads = m_threads.get(algorithm);

    if (threads.isEmpty()) {
        return out;
    }

    out.reserve(threads.count() * 2);

    for (const auto &thread : threads.data()) {
        if (thread.index() >= devices.size()) {
            LOG_INFO("%s" YELLOW(" skip non-existing device with index ") YELLOW_BOLD("%u"), ocl_tag(), thread.index());
            continue;
        }

        if (thread.threads().size() > 1) {
            for (int64_t affinity : thread.threads()) {
                out.emplace_back(miner, algorithm, *this, platform, thread, devices[thread.index()], affinity);
            }
        }
        else {
            out.emplace_back(miner, algorithm, *this, platform, thread, devices[thread.index()], thread.threads().front());
        }
    }

    return out;
}


void xmrig::OclConfig::read(const rapidjson::Value &value)
{
    if (value.IsObject()) {
        m_enabled   = Json::getBool(value, kEnabled, m_enabled);
        m_cache     = Json::getBool(value, kCache, m_cache);
        m_loader    = Json::getString(value, kLoader);

#       ifndef XMRIG_OS_APPLE
        setPlatform(Json::getValue(value, kPlatform));
#       endif

        setDevicesHint(Json::getString(value, kDevicesHint));

#       ifdef XMRIG_FEATURE_ADL
        m_adl = Json::getBool(value, kAdl, m_adl);
#       endif

        m_threads.read(value);

        generate();
    }
    else if (value.IsBool()) {
        m_enabled = value.GetBool();

        generate();
    }
    else {
        m_shouldSave = true;

        generate();
    }
}


void xmrig::OclConfig::generate()
{
    if (!isEnabled() || m_threads.has("*")) {
        return;
    }

    if (!OclLib::init(loader())) {
        return;
    }

    const auto devices = m_devicesHint.empty() ? platform().devices() : filterDevices(platform().devices(), m_devicesHint);
    if (devices.empty()) {
        return;
    }

    size_t count = 0;

    count += xmrig::generate<Algorithm::CN>(m_threads, devices);
    count += xmrig::generate<Algorithm::CN_LITE>(m_threads, devices);
    count += xmrig::generate<Algorithm::CN_HEAVY>(m_threads, devices);
    count += xmrig::generate<Algorithm::CN_PICO>(m_threads, devices);
    count += xmrig::generate<Algorithm::CN_FEMTO>(m_threads, devices);
    count += xmrig::generate<Algorithm::RANDOM_X>(m_threads, devices);
    count += xmrig::generate<Algorithm::KAWPOW>(m_threads, devices);

    m_shouldSave = count > 0;
}


void xmrig::OclConfig::setDevicesHint(const char *devicesHint)
{
    if (devicesHint == nullptr) {
        return;
    }

    const auto indexes = String(devicesHint).split(',');
    m_devicesHint.reserve(indexes.size());

    for (const auto &index : indexes) {
        m_devicesHint.push_back(strtoul(index, nullptr, 10));
    }
}


#ifndef XMRIG_OS_APPLE
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
#endif
