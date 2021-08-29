/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <cassert>
#include <cstring>


#ifdef _MSC_VER
#   define strcasecmp  _stricmp
#endif


#include "crypto/common/Assembly.h"
#include "3rdparty/rapidjson/document.h"


namespace xmrig {


static const char *asmNames[] = {
    "none",
    "auto",
    "intel",
    "ryzen",
    "bulldozer"
};


} /* namespace xmrig */


xmrig::Assembly::Id xmrig::Assembly::parse(const char *assembly, Id defaultValue)
{
    constexpr size_t const size = sizeof(asmNames) / sizeof((asmNames)[0]);
    static_assert(size == MAX, "asmNames size mismatch");

    if (assembly == nullptr) {
        return defaultValue;
    }

    for (size_t i = 0; i < size; i++) {
        if (strcasecmp(assembly, asmNames[i]) == 0) {
            return static_cast<Id>(i);
        }
    }

    return defaultValue;
}


xmrig::Assembly::Id xmrig::Assembly::parse(const rapidjson::Value &value, Id defaultValue)
{
    if (value.IsBool()) {
        return value.GetBool() ? AUTO : NONE;
    }

    if (value.IsString()) {
        return parse(value.GetString(), defaultValue);
    }

    return defaultValue;
}


const char *xmrig::Assembly::toString() const
{
    return asmNames[m_id];
}


rapidjson::Value xmrig::Assembly::toJSON() const
{
    using namespace rapidjson;

    if (m_id == NONE) {
        return Value(false);
    }

    if (m_id == AUTO) {
        return Value(true);
    }

    return Value(StringRef(toString()));
}
