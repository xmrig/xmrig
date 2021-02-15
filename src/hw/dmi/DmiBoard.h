/* xmlcore
 * Copyright (c) 2000-2002 Alan Cox     <alan@redhat.com>
 * Copyright (c) 2005-2020 Jean Delvare <jdelvare@suse.de>
 * Copyright (c) 2018-2021 SChernykh    <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 xmlcore        <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_DMIBOARD_H
#define xmlcore_DMIBOARD_H


#include "base/tools/String.h"


namespace xmlcore {


struct dmi_header;


class DmiBoard
{
public:
    DmiBoard() = default;

    inline const String &product() const    { return m_product; }
    inline const String &vendor() const     { return m_vendor; }
    inline bool isValid() const             { return !m_product.isEmpty() && !m_vendor.isEmpty(); }

    void decode(dmi_header *h);

#   ifdef xmlcore_FEATURE_API
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
#   endif

private:
    String m_product;
    String m_vendor;
};


} /* namespace xmlcore */


#endif /* xmlcore_DMIBOARD_H */
