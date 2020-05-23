/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#ifndef XMRIG_ASSEMBLY_H
#define XMRIG_ASSEMBLY_H


#include "3rdparty/rapidjson/fwd.h"


namespace xmrig {


class Assembly
{
public:
    enum Id : int {
        NONE,
        AUTO,
        INTEL,
        RYZEN,
        BULLDOZER,
        MAX
    };


    inline Assembly()                                                   {}
    inline Assembly(Id id) : m_id(id)                                   {}
    inline Assembly(const char *assembly) : m_id(parse(assembly))       {}
    inline Assembly(const rapidjson::Value &value) : m_id(parse(value)) {}

    static Id parse(const char *assembly, Id defaultValue = AUTO);
    static Id parse(const rapidjson::Value &value, Id defaultValue = AUTO);

    const char *toString() const;
    rapidjson::Value toJSON() const;

    inline bool isEqual(const Assembly &other) const      { return m_id == other.m_id; }

    inline bool operator!=(Assembly::Id id) const         { return m_id != id; }
    inline bool operator!=(const Assembly &other) const   { return !isEqual(other); }
    inline bool operator==(Assembly::Id id) const         { return m_id == id; }
    inline bool operator==(const Assembly &other) const   { return isEqual(other); }
    inline operator Assembly::Id() const                  { return m_id; }

private:
    Id m_id = AUTO;
};


} /* namespace xmrig */


#endif /* XMRIG_ASSEMBLY_H */
