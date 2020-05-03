/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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

#ifndef XMRIG_COIN_H
#define XMRIG_COIN_H


#include "3rdparty/rapidjson/fwd.h"
#include "base/crypto/Algorithm.h"


namespace xmrig {


class Coin
{
public:
    enum Id : int {
        INVALID = -1,
        MONERO,
        ARQMA,
        DERO,
        KEVA
    };


    Coin() = default;
    inline Coin(const char *name) : m_id(parse(name)) {}
    inline Coin(Id id) : m_id(id)                     {}


    inline bool isEqual(const Coin &other) const        { return m_id == other.m_id; }
    inline bool isValid() const                         { return m_id != INVALID; }
    inline Id id() const                                { return m_id; }

    Algorithm::Id algorithm(uint8_t blobVersion) const;
    const char *name() const;
    rapidjson::Value toJSON() const;

    inline bool operator!=(Coin::Id id) const           { return m_id != id; }
    inline bool operator!=(const Coin &other) const     { return !isEqual(other); }
    inline bool operator==(Coin::Id id) const           { return m_id == id; }
    inline bool operator==(const Coin &other) const     { return isEqual(other); }
    inline operator Coin::Id() const                    { return m_id; }

    static Id parse(const char *name);

private:
    Id m_id = INVALID;
};


} /* namespace xmrig */


#endif /* XMRIG_COIN_H */
