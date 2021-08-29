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

#ifndef XMRIG_MSRITEM_H
#define XMRIG_MSRITEM_H


#include "base/tools/String.h"


#include <limits>
#include <vector>


namespace xmrig
{


class MsrItem
{
public:
    constexpr static uint64_t kNoMask = std::numeric_limits<uint64_t>::max();

    inline MsrItem() = default;
    inline MsrItem(uint32_t reg, uint64_t value, uint64_t mask = kNoMask) : m_reg(reg), m_value(value), m_mask(mask) {}

    MsrItem(const rapidjson::Value &value);

    inline bool isValid() const     { return m_reg > 0; }
    inline uint32_t reg() const     { return m_reg; }
    inline uint64_t value() const   { return m_value; }
    inline uint64_t mask() const    { return m_mask; }

    static inline uint64_t maskedValue(uint64_t old_value, uint64_t new_value, uint64_t mask)
    {
        return (new_value & mask) | (old_value & ~mask);
    }

    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    String toString() const;

private:
    uint32_t m_reg      = 0;
    uint64_t m_value    = 0;
    uint64_t m_mask     = kNoMask;
};


using MsrItems = std::vector<MsrItem>;


} /* namespace xmrig */


#endif /* XMRIG_MSRITEM_H */
