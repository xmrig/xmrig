/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include <assert.h>
#include <string.h>


#ifdef _MSC_VER
#   define strncasecmp _strnicmp
#   define strcasecmp  _stricmp
#endif


#include "crypto/Asm.h"
#include "rapidjson/document.h"


static const char *asmNames[] = {
    "none",
    "auto",
    "intel",
    "ryzen"
};


xmrig::Assembly xmrig::Asm::parse(const char *assembly, Assembly defaultValue)
{
    constexpr size_t const size = sizeof(asmNames) / sizeof((asmNames)[0]);
    assert(assembly != nullptr);
    assert(ASM_MAX == size);

    if (assembly == nullptr) {
        return defaultValue;
    }

    for (size_t i = 0; i < size; i++) {
        if (strcasecmp(assembly, asmNames[i]) == 0) {
            return static_cast<Assembly>(i);
        }
    }

    return defaultValue;
}


xmrig::Assembly xmrig::Asm::parse(const rapidjson::Value &value, Assembly defaultValue)
{
    if (value.IsBool()) {
        return parse(value.GetBool());
    }

    if (value.IsString()) {
        return parse(value.GetString(), defaultValue);
    }

    return defaultValue;
}


const char *xmrig::Asm::toString(Assembly assembly)
{
    return asmNames[assembly];
}


rapidjson::Value xmrig::Asm::toJSON(Assembly assembly)
{
    using namespace rapidjson;

    if (assembly == ASM_NONE) {
        return Value(false);
    }

    if (assembly == ASM_AUTO) {
        return Value(true);
    }

    return Value(StringRef(toString(assembly)));
}
