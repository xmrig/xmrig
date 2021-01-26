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


#include "hw/dmi/DmiBoard.h"
#include "3rdparty/rapidjson/document.h"
#include "hw/dmi/DmiTools.h"


void xmrig::DmiBoard::decode(dmi_header *h)
{
    if (h->length < 0x08) {
        return;
    }

    m_vendor  = dmi_string(h, 0x04);
    m_product = dmi_string(h, 0x05);
}


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::DmiBoard::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    auto &allocator = doc.GetAllocator();
    Value out(kObjectType);
    out.AddMember("vendor",     m_vendor.toJSON(doc), allocator);
    out.AddMember("product",    m_product.toJSON(doc), allocator);

    return out;
}
#endif
