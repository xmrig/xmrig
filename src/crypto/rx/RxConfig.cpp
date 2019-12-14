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


#include "crypto/rx/RxConfig.h"
#include "backend/cpu/Cpu.h"
#include "base/io/json/Json.h"
#include "rapidjson/document.h"


#ifdef XMRIG_FEATURE_HWLOC
#   include "backend/cpu/platform/HwlocCpuInfo.h"
#endif


#include <array>
#include <algorithm>
#include <cmath>


#ifdef _MSC_VER
#   define strcasecmp  _stricmp
#endif


namespace xmrig {

static const char *kInit        = "init";
static const char *kMode        = "mode";
static const char *kOneGbPages  = "1gb-pages";
static const char *kWrmsr       = "wrmsr";

#ifdef XMRIG_FEATURE_HWLOC
static const char *kNUMA        = "numa";
#endif

static const std::array<const char *, RxConfig::ModeMax> modeNames = { "auto", "fast", "light" };

}


bool xmrig::RxConfig::read(const rapidjson::Value &value)
{
    if (value.IsObject()) {
        m_threads    = Json::getInt(value, kInit, m_threads);
        m_mode       = readMode(Json::getValue(value, kMode));
        m_wrmsr      = readMSR(Json::getValue(value, kWrmsr));

#       ifdef XMRIG_OS_LINUX
        m_oneGbPages = Json::getBool(value, kOneGbPages, m_oneGbPages);
#       endif

#       ifdef XMRIG_FEATURE_HWLOC
        if (m_mode == LightMode) {
            m_numa = false;

            return true;
        }

        const auto &numa = Json::getValue(value, kNUMA);
        if (numa.IsArray()) {
            m_nodeset.reserve(numa.Size());

            for (const auto &node : numa.GetArray()) {
                if (node.IsUint()) {
                    m_nodeset.emplace_back(node.GetUint());
                }
            }
        }
        else if (numa.IsBool()) {
            m_numa = numa.GetBool();
        }
#       endif

        return true;
    }

    return false;
}


rapidjson::Value xmrig::RxConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value obj(kObjectType);
    obj.AddMember(StringRef(kInit),         m_threads, allocator);
    obj.AddMember(StringRef(kMode),         StringRef(modeName()), allocator);
    obj.AddMember(StringRef(kOneGbPages),   m_oneGbPages, allocator);

    if (m_wrmsr < 0 || m_wrmsr == 6) {
        obj.AddMember(StringRef(kWrmsr), m_wrmsr == 6, allocator);
    }
    else {
        obj.AddMember(StringRef(kWrmsr), m_wrmsr, allocator);
    }

#   ifdef XMRIG_FEATURE_HWLOC
    if (!m_nodeset.empty()) {
        Value numa(kArrayType);

        for (uint32_t i : m_nodeset) {
            numa.PushBack(i, allocator);
        }

        obj.AddMember(StringRef(kNUMA), numa, allocator);
    }
    else {
        obj.AddMember(StringRef(kNUMA), m_numa, allocator);
    }
#   endif

    return obj;
}


#ifdef XMRIG_FEATURE_HWLOC
std::vector<uint32_t> xmrig::RxConfig::nodeset() const
{
    if (!m_nodeset.empty()) {
        return m_nodeset;
    }

    return (m_numa && Cpu::info()->nodes() > 1) ? static_cast<HwlocCpuInfo *>(Cpu::info())->nodeset() : std::vector<uint32_t>();
}
#endif


const char *xmrig::RxConfig::modeName() const
{
    return modeNames[m_mode];
}


uint32_t xmrig::RxConfig::threads(uint32_t limit) const
{
    if (m_threads > 0) {
        return m_threads;
    }

    if (limit < 100) {
        return std::max(static_cast<uint32_t>(round(Cpu::info()->threads() * (limit / 100.0))), 1U);
    }

    return Cpu::info()->threads();
}


int xmrig::RxConfig::readMSR(const rapidjson::Value &value) const
{
    if (value.IsInt()) {
        return std::min(value.GetInt(), 15);
    }

    if (value.IsBool() && !value.GetBool()) {
        return -1;
    }

    return m_wrmsr;
}


xmrig::RxConfig::Mode xmrig::RxConfig::readMode(const rapidjson::Value &value) const
{
    if (value.IsUint()) {
        return static_cast<Mode>(std::min(value.GetUint(), ModeMax - 1));
    }

    if (value.IsString()) {
        auto mode = value.GetString();

        for (size_t i = 0; i < modeNames.size(); i++) {
            if (strcasecmp(mode, modeNames[i]) == 0) {
                return static_cast<Mode>(i);
            }
        }
    }

    return AutoMode;
}
