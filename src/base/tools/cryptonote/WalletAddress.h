/* XMRig
 * Copyright (c) 2012-2013 The Cryptonote developers
 * Copyright (c) 2014-2021 The Monero Project
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

#ifndef XMRIG_WALLETADDRESS_H
#define XMRIG_WALLETADDRESS_H


#include "base/tools/String.h"
#include "base/crypto/Coin.h"


namespace xmrig {


class WalletAddress
{
public:
    enum Net : uint32_t {
        MAINNET,
        TESTNET,
        STAGENET
    };

    enum Type : uint32_t {
        PUBLIC,
        INTEGRATED,
        SUBADDRESS
    };

    constexpr static size_t kKeySize        = 32;
    constexpr static size_t kMaxSize        = 256;
    constexpr static size_t kMinDataSize    = 69;
    constexpr static size_t kMinSize        = 95;

    WalletAddress() = default;
    inline WalletAddress(const char *address, size_t size)  { decode(address, size); }
    inline WalletAddress(const char *address)               { decode(address); }
    inline WalletAddress(const rapidjson::Value &address)   { decode(address); }
    inline WalletAddress(const String &address)             { decode(address); }

    inline bool decode(const char *address)                 { return decode(address, strlen(address)); }
    inline bool decode(const String &address)               { return decode(address, address.size()); }
    inline bool isValid() const                             { return m_tag > 0 && m_data.size() >= kMinSize; }
    inline const char *data() const                         { return m_data; }
    inline const Coin &coin() const                         { return tagInfo(m_tag).coin; }
    inline const uint8_t *spendKey() const                  { return m_publicSpendKey; }
    inline const uint8_t *viewKey() const                   { return m_publicViewKey; }
    inline Net net() const                                  { return tagInfo(m_tag).net; }
    inline Type type() const                                { return tagInfo(m_tag).type; }
    inline uint16_t rpcPort() const                         { return tagInfo(m_tag).rpcPort; }
    inline uint16_t zmqPort() const                         { return tagInfo(m_tag).zmqPort; }
    inline uint64_t tag() const                             { return m_tag; }

    bool decode(const char *address, size_t size);
    bool decode(const rapidjson::Value &address);
    const char *netName() const;
    const char *typeName() const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;

#   ifdef XMRIG_FEATURE_API
    rapidjson::Value toAPI(rapidjson::Document &doc) const;
#   endif

private:
    struct TagInfo
    {
        const Coin coin;
        const Net net;
        const Type type;
        const uint16_t rpcPort;
        const uint16_t zmqPort;
    };

    static const TagInfo &tagInfo(uint64_t tag);

    String m_data;
    uint64_t m_tag  = 0;
    uint8_t m_checksum[4]{};
    uint8_t m_publicSpendKey[kKeySize]{};
    uint8_t m_publicViewKey[kKeySize]{};
};


} /* namespace xmrig */


#endif /* XMRIG_WALLETADDRESS_H */
