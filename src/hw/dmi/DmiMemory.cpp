/* XMRig
 * Copyright (c) 2000-2002 Alan Cox     <alan@redhat.com>
 * Copyright (c) 2005-2020 Jean Delvare <jdelvare@suse.de>
 * Copyright (c) 2018-2021 SChernykh    <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig        <https://github.com/xmrig>, <support@xmrig.com>
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


#include "hw/dmi/DmiMemory.h"
#include "3rdparty/fmt/format.h"
#include "3rdparty/rapidjson/document.h"
#include "hw/dmi/DmiTools.h"


#include <algorithm>
#include <array>
#include <regex>


namespace xmrig {


static const char *kIdFormat = "DIMM_{}{}";


static inline uint16_t dmi_memory_device_width(uint16_t code)
{
    return (code == 0xFFFF || code == 0) ? 0 : code;
}


static const char *dmi_memory_device_form_factor(uint8_t code)
{
    static const std::array<const char *, 0x10> form_factor
    {
        "Other",
        "Unknown",
        "SIMM",
        "SIP",
        "Chip",
        "DIP",
        "ZIP",
        "Proprietary Card",
        "DIMM",
        "TSOP",
        "Row Of Chips",
        "RIMM",
        "SODIMM",
        "SRIMM",
        "FB-DIMM",
        "Die"
    };

    if (code >= 0x01 && code <= form_factor.size()) {
        return form_factor[code - 0x01];
    }

    return form_factor[1];
}


static const char *dmi_memory_device_type(uint8_t code)
{
    static const std::array<const char *, 0x23> type
    {
        "Other", /* 0x01 */
        "Unknown",
        "DRAM",
        "EDRAM",
        "VRAM",
        "SRAM",
        "RAM",
        "ROM",
        "Flash",
        "EEPROM",
        "FEPROM",
        "EPROM",
        "CDRAM",
        "3DRAM",
        "SDRAM",
        "SGRAM",
        "RDRAM",
        "DDR",
        "DDR2",
        "DDR2 FB-DIMM",
        "Reserved",
        "Reserved",
        "Reserved",
        "DDR3",
        "FBD2",
        "DDR4",
        "LPDDR",
        "LPDDR2",
        "LPDDR3",
        "LPDDR4",
        "Logical non-volatile device",
        "HBM",
        "HBM2",
        "DDR5",
        "LPDDR5"
    };

    if (code >= 0x01 && code <= type.size()) {
        return type[code - 0x01];
    }

    return type[1];
}


static uint64_t dmi_memory_device_speed(uint16_t code1, uint32_t code2)
{
    return (code1 == 0xFFFF) ? code2 : code1;
}


} // namespace xmrig



xmrig::DmiMemory::DmiMemory(dmi_header *h)
{
    if (h->length < 0x15) {
        return;
    }

    m_totalWidth = dmi_memory_device_width(dmi_get<uint16_t>(h, 0x08));
    m_width      = dmi_memory_device_width(dmi_get<uint16_t>(h, 0x0A));

    auto size = dmi_get<uint16_t>(h, 0x0C);
    if (h->length >= 0x20 && size == 0x7FFF) {
        m_size = (dmi_get<uint32_t>(h, 0x1C) & 0x7FFFFFFFUL) * 1024ULL * 1024ULL;
    }
    else if (size) {
        m_size = (1024ULL * (size & 0x7FFF) * ((size & 0x8000) ? 1 : 1024ULL));
    }

    setId(dmi_string(h, 0x10), dmi_string(h, 0x11));

    m_formFactor = h->data[0x0E];
    m_type       = h->data[0x12];

    if (!m_size || h->length < 0x17) {
        return;
    }

    m_speed = dmi_memory_device_speed(dmi_get<uint16_t>(h, 0x15), h->length >= 0x5C ? dmi_get<uint32_t>(h, 0x54) : 0) * 1000000ULL;

    if (h->length < 0x1B) {
        return;
    }

    m_vendor  = dmi_string(h, 0x17);
    m_product = dmi_string(h, 0x1A);

    if (h->length < 0x1C) {
        return;
    }

    m_rank = h->data[0x1B] & 0x0F;

    if (h->length < 0x22) {
        return;
    }

    const uint64_t configuredSpeed = dmi_memory_device_speed(dmi_get<uint16_t>(h, 0x20), h->length >= 0x5C ? dmi_get<uint32_t>(h, 0x58) : 0) * 1000000ULL;
    m_speed = configuredSpeed ? configuredSpeed : m_speed;

    if (h->length < 0x28) {
        return;
    }

    m_voltage = dmi_get<uint16_t>(h, 0x26);
}


const char *xmrig::DmiMemory::formFactor() const
{
    return dmi_memory_device_form_factor(m_formFactor);
}


const char *xmrig::DmiMemory::type() const
{
    return dmi_memory_device_type(m_type);
}


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::DmiMemory::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    auto &allocator = doc.GetAllocator();
    Value out(kObjectType);
    out.AddMember("id",             id().toJSON(doc), allocator);
    out.AddMember("slot",           m_slot.toJSON(doc), allocator);
    out.AddMember("type",           StringRef(type()), allocator);
    out.AddMember("form_factor",    StringRef(formFactor()), allocator);
    out.AddMember("size",           m_size, allocator);
    out.AddMember("speed",          m_speed, allocator);
    out.AddMember("rank",           m_rank, allocator);
    out.AddMember("voltage",        m_voltage, allocator);
    out.AddMember("width",          m_width, allocator);
    out.AddMember("total_width",    m_totalWidth, allocator);
    out.AddMember("vendor",         m_vendor.toJSON(doc), allocator);
    out.AddMember("product",        m_product.toJSON(doc), allocator);
    out.AddMember("bank",           m_bank.toJSON(doc), allocator);

    return out;
}
#endif


void xmrig::DmiMemory::setId(const char *slot, const char *bank)
{
    m_slot = slot;
    m_bank = bank;

    try {
        std::cmatch cm;
        if (std::regex_match(slot, cm, std::regex("^Channel([A-Z])[-_]DIMM(\\d+)$", std::regex_constants::icase))) {
            m_id = fmt::format(kIdFormat, cm.str(1), cm.str(2)).c_str();
        }
        else if (std::regex_search(bank, cm, std::regex("CHANNEL ([A-Z])$"))) {
            std::cmatch cm2;
            if (std::regex_match(slot, cm2, std::regex("^DIMM (\\d+)$"))) {
                m_id = fmt::format(kIdFormat, cm.str(1), cm2.str(1)).c_str();
            }
        }
    } catch (...) {}
}
