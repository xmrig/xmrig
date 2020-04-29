/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
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


#include "crypto/rx/msr/MsrItem.h"
#include "3rdparty/rapidjson/document.h"


#include <cstdio>


xmrig::MsrItem::MsrItem(const rapidjson::Value &value)
{
    if (!value.IsString()) {
        return;
    }

    auto kv = String(value.GetString()).split(':');
    if (kv.size() < 2) {
        return;
    }

    m_reg   = strtoul(kv[0], nullptr, 0);
    m_value = strtoull(kv[1], nullptr, 0);
    m_mask  = (kv.size() > 2) ? strtoull(kv[2], nullptr, 0) : kNoMask;
}


rapidjson::Value xmrig::MsrItem::toJSON(rapidjson::Document &doc) const
{
    return toString().toJSON(doc);
}


xmrig::String xmrig::MsrItem::toString() const
{
    constexpr size_t size = 48;

    auto buf = new char[size]();

    if (m_mask != kNoMask) {
        snprintf(buf, size, "0x%" PRIx32 ":0x%" PRIx64 ":0x%" PRIx64, m_reg, m_value, m_mask);
    }
    else {
        snprintf(buf, size, "0x%" PRIx32 ":0x%" PRIx64, m_reg, m_value);
    }

    return buf;
}
