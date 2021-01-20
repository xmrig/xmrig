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

#ifndef XMRIG_DMIMEMORY_H
#define XMRIG_DMIMEMORY_H


#include "base/tools/String.h"


namespace xmrig {


struct dmi_header;


class DmiMemory
{
public:
    DmiMemory() = default;
    DmiMemory(dmi_header *h);

    inline bool isValid() const             { return !m_slot.isEmpty(); }
    inline const String &bank() const       { return m_bank; }
    inline const String &product() const    { return m_product; }
    inline const String &slot() const       { return m_slot; }
    inline const String &vendor() const     { return m_vendor; }
    inline uint16_t totalWidth() const      { return m_totalWidth; }
    inline uint16_t voltage() const         { return m_voltage; }
    inline uint16_t width() const           { return m_width; }
    inline uint64_t size() const            { return m_size; }
    inline uint64_t speed() const           { return m_speed; }
    inline uint8_t rank() const             { return m_rank; }

    const char *formFactor() const;
    const char *type() const;

#   ifdef XMRIG_FEATURE_API
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
#   endif

private:
    String m_bank;
    String m_product;
    String m_slot;
    String m_vendor;
    uint16_t m_totalWidth   = 0;
    uint16_t m_voltage      = 0;
    uint16_t m_width        = 0;
    uint64_t m_size         = 0;
    uint64_t m_speed        = 0;
    uint8_t m_formFactor    = 0;
    uint8_t m_rank          = 0;
    uint8_t m_type          = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_DMIMEMORY_H */
