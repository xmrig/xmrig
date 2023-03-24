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

#include "crypto/rx/RxConfig.h"
#include "3rdparty/rapidjson/document.h"
#include "backend/cpu/Cpu.h"
#include "base/io/json/Json.h"


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

const char *RxConfig::kInit                     = "init";
const char *RxConfig::kInitAVX2                 = "init-avx2";
const char *RxConfig::kField                    = "randomx";
const char *RxConfig::kMode                     = "mode";
const char *RxConfig::kOneGbPages               = "1gb-pages";
const char *RxConfig::kRdmsr                    = "rdmsr";
const char *RxConfig::kWrmsr                    = "wrmsr";
const char *RxConfig::kScratchpadPrefetchMode   = "scratchpad_prefetch_mode";
const char *RxConfig::kCacheQoS                 = "cache_qos";

#ifdef XMRIG_FEATURE_HWLOC
const char *RxConfig::kNUMA                     = "numa";
#endif


static const std::array<const char *, RxConfig::ModeMax> modeNames = { "auto", "fast", "light" };


#ifdef XMRIG_FEATURE_MSR
constexpr size_t kMsrArraySize = 6;

static const std::array<MsrItems, kMsrArraySize> msrPresets = {
    MsrItems(),
    MsrItems{{ 0xC0011020, 0ULL }, { 0xC0011021, 0x40ULL, ~0x20ULL }, { 0xC0011022, 0x1510000ULL }, { 0xC001102b, 0x2000cc16ULL }},
    MsrItems{{ 0xC0011020, 0x0004480000000000ULL }, { 0xC0011021, 0x001c000200000040ULL, ~0x20ULL }, { 0xC0011022, 0xc000000401570000ULL }, { 0xC001102b, 0x2000cc10ULL }},
    MsrItems{{ 0xC0011020, 0x0004400000000000ULL }, { 0xC0011021, 0x0004000000000040ULL, ~0x20ULL }, { 0xC0011022, 0x8680000401570000ULL }, { 0xC001102b, 0x2040cc10ULL }},
    MsrItems{{ 0x1a4, 0xf }},
    MsrItems()
};

static const std::array<const char *, kMsrArraySize> modNames = { MSR_NAMES_LIST };

static_assert (kMsrArraySize == ICpuInfo::MSR_MOD_MAX, "kMsrArraySize and MSR_MOD_MAX mismatch");
#endif


} // namespace xmrig


bool xmrig::RxConfig::read(const rapidjson::Value &value)
{
    if (value.IsObject()) {
        m_threads         = Json::getInt(value, kInit, m_threads);
        m_initDatasetAVX2 = Json::getInt(value, kInitAVX2, m_initDatasetAVX2);
        m_mode            = readMode(Json::getValue(value, kMode));
        m_rdmsr           = Json::getBool(value, kRdmsr, m_rdmsr);

#       ifdef XMRIG_FEATURE_MSR
        readMSR(Json::getValue(value, kWrmsr));
#       endif

        m_cacheQoS = Json::getBool(value, kCacheQoS, m_cacheQoS);

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

        const auto mode = static_cast<uint32_t>(Json::getInt(value, kScratchpadPrefetchMode, static_cast<int>(m_scratchpadPrefetchMode)));
        if (mode < ScratchpadPrefetchMax) {
            m_scratchpadPrefetchMode = static_cast<ScratchpadPrefetchMode>(mode);
        }

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
    obj.AddMember(StringRef(kInitAVX2),     m_initDatasetAVX2, allocator);
    obj.AddMember(StringRef(kMode),         StringRef(modeName()), allocator);
    obj.AddMember(StringRef(kOneGbPages),   m_oneGbPages, allocator);
    obj.AddMember(StringRef(kRdmsr),        m_rdmsr, allocator);

#   ifdef XMRIG_FEATURE_MSR
    if (!m_msrPreset.empty()) {
        Value wrmsr(kArrayType);
        wrmsr.Reserve(m_msrPreset.size(), allocator);

        for (const auto &i : m_msrPreset) {
            wrmsr.PushBack(i.toJSON(doc), allocator);
        }

        obj.AddMember(StringRef(kWrmsr), wrmsr, allocator);
    }
    else {
        obj.AddMember(StringRef(kWrmsr), m_wrmsr, allocator);
    }
#   else
    obj.AddMember(StringRef(kWrmsr), false, allocator);
#   endif

    obj.AddMember(StringRef(kCacheQoS), m_cacheQoS, allocator);

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

    obj.AddMember(StringRef(kScratchpadPrefetchMode), static_cast<int>(m_scratchpadPrefetchMode), allocator);

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


#ifdef XMRIG_FEATURE_MSR
const char *xmrig::RxConfig::msrPresetName() const
{
    return modNames[msrMod()];
}


const xmrig::MsrItems &xmrig::RxConfig::msrPreset() const
{
    const auto mod = msrMod();

    if (mod == ICpuInfo::MSR_MOD_CUSTOM) {
        return m_msrPreset;
    }

    return msrPresets[mod];
}


uint32_t xmrig::RxConfig::msrMod() const
{
    if (!wrmsr()) {
        return ICpuInfo::MSR_MOD_NONE;
    }

    if (!m_msrPreset.empty()) {
        return ICpuInfo::MSR_MOD_CUSTOM;
    }

    return Cpu::info()->msrMod();
}


void xmrig::RxConfig::readMSR(const rapidjson::Value &value)
{
    if (value.IsBool()) {
        m_wrmsr = value.GetBool();

        return;
    }

    if (value.IsInt()) {
        const int i = std::min(value.GetInt(), 15);
        if (i >= 0) {
            if (Cpu::info()->vendor() == ICpuInfo::VENDOR_INTEL) {
                m_msrPreset.emplace_back(0x1a4, i);
            }
        }
        else {
            m_wrmsr = false;
        }
    }

    if (value.IsArray()) {
        for (const auto &i : value.GetArray()) {
            MsrItem item(i);
            if (item.isValid()) {
                m_msrPreset.emplace_back(item);
            }
        }

        m_wrmsr = !m_msrPreset.empty();
    }
}
#endif


xmrig::RxConfig::Mode xmrig::RxConfig::readMode(const rapidjson::Value &value)
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
