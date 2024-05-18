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

#ifndef XMRIG_COIN_H
#define XMRIG_COIN_H


#include "3rdparty/rapidjson/fwd.h"
#include "base/crypto/Algorithm.h"


namespace xmrig {


class Coin
{
public:
    enum Id : uint32_t {
        INVALID,
        MONERO,
        SUMO,
        ARQ,
        XEQ,
        GRAFT,
        KEVA,
        RAVEN,
        WOWNERO,
        ZEPHYR,
        TOWNFORGE,
        MAX
    };

    static const char *kDisabled;
    static const char *kField;
    static const char *kUnknown;

    Coin() = default;
    Coin(const rapidjson::Value &value);
    inline Coin(const char *name) : m_id(parse(name))                           {}
    inline Coin(Id id) : m_id(id)                                               {}
    inline Coin(uint32_t id) : m_id(id < MAX ? static_cast<Id>(id) : INVALID)   {}


    inline bool isEqual(const Coin &other) const                                { return m_id == other.m_id; }
    inline bool isValid() const                                                 { return m_id != INVALID; }
    inline Id id() const                                                        { return m_id; }
    inline const char *tag() const                                              { return tag(m_id); }
    inline double decimal(uint64_t amount) const                                { return static_cast<double>(amount) / units(); }

    Algorithm algorithm(uint8_t blobVersion = 255) const;
    const char *code() const;
    const char *name() const;
    rapidjson::Value toJSON() const;
    uint64_t target(uint8_t blobVersion = 255) const;
    uint64_t units() const;

    inline bool operator!=(Id id) const                                         { return m_id != id; }
    inline bool operator!=(const Coin &other) const                             { return !isEqual(other); }
    inline bool operator<(Id id) const                                          { return m_id < id; }
    inline bool operator<(const Coin &other) const                              { return m_id < other.m_id; }
    inline bool operator==(Id id) const                                         { return m_id == id; }
    inline bool operator==(const Coin &other) const                             { return isEqual(other); }
    inline operator Id() const                                                  { return m_id; }

    static Id parse(const char *name);
    static const char *tag(Id id);

private:
    Id m_id = INVALID;
};


} /* namespace xmrig */


#endif /* XMRIG_COIN_H */
