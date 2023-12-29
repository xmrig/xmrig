/* XMRig
 * Copyright (c) 2012-2013 The Cryptonote developers
 * Copyright (c) 2014-2021 The Monero Project
 * Copyright (c) 2018-2023 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2023 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "base/tools/cryptonote/WalletAddress.h"
#include "3rdparty/rapidjson/document.h"
#include "base/crypto/keccak.h"
#include "base/tools/Buffer.h"
#include "base/tools/cryptonote/BlobReader.h"
#include "base/tools/cryptonote/umul128.h"
#include "base/tools/Cvt.h"


#include <array>
#include <map>


bool xmrig::WalletAddress::decode(const char *address, size_t size)
{
    uint64_t tf_tag = 0;
    if (size >= 4 && !strncmp(address, "TF", 2)) {
      tf_tag = 0x424200;
      switch (address[2])
      {
        case '1': tf_tag |= 0; break;
        case '2': tf_tag |= 1; break;
        default: tf_tag = 0; return false;
      }
      switch (address[3]) {
        case 'M': tf_tag |= 0; break;
        case 'T': tf_tag |= 0x10; break;
        case 'S': tf_tag |= 0x20; break;
        default: tf_tag = 0; return false;
      }
      address += 4;
      size -= 4;
    }

    static constexpr std::array<int, 9> block_sizes{ 0, 2, 3, 5, 6, 7, 9, 10, 11 };
    static constexpr char alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    constexpr size_t alphabet_size = sizeof(alphabet) - 1;

    if (size < kMinSize || size > kMaxSize) {
        return false;
    }

    int8_t reverse_alphabet[256];
    memset(reverse_alphabet, -1, sizeof(reverse_alphabet));

    for (size_t i = 0; i < alphabet_size; ++i) {
        reverse_alphabet[static_cast<int>(alphabet[i])] = i;
    }

    const int len = static_cast<int>(size);
    const int num_full_blocks = len / block_sizes.back();
    const int last_block_size = len % block_sizes.back();

    int last_block_size_index = -1;

    for (size_t i = 0; i < block_sizes.size(); ++i) {
        if (block_sizes[i] == last_block_size) {
            last_block_size_index = i;
            break;
        }
    }

    if (last_block_size_index < 0) {
        return false;
    }

    const size_t data_size = static_cast<size_t>(num_full_blocks) * sizeof(uint64_t) + last_block_size_index;
    if (data_size < kMinDataSize) {
        return false;
    }

    Buffer data;
    data.reserve(data_size);

    const char *address_data = address;

    for (int i = 0; i <= num_full_blocks; ++i) {
        uint64_t num = 0;
        uint64_t order = 1;

        for (int j = ((i < num_full_blocks) ? block_sizes.back() : last_block_size) - 1; j >= 0; --j) {
            const int digit = reverse_alphabet[static_cast<int>(address_data[j])];
            if (digit < 0) {
                return false;
            }

            uint64_t hi;
            const uint64_t tmp = num + __umul128(order, static_cast<uint64_t>(digit), &hi);
            if ((tmp < num) || hi) {
                return false;
            }

            num = tmp;
            order *= alphabet_size;
        }

        address_data += block_sizes.back();

        auto p = reinterpret_cast<const uint8_t*>(&num);
        for (int j = ((i < num_full_blocks) ? static_cast<int>(sizeof(num)) : last_block_size_index) - 1; j >= 0; --j) {
            data.emplace_back(p[j]);
        }
    }

    assert(data.size() == data_size);

    BlobReader<false> ar(data.data(), data_size);

    if (ar(m_tag) && ar(m_publicSpendKey) && ar(m_publicViewKey) && ar.skip(ar.remaining() - sizeof(m_checksum)) && ar(m_checksum)) {
        uint8_t md[200];
        keccak(data.data(), data_size - sizeof(m_checksum), md);

        if (memcmp(m_checksum, md, sizeof(m_checksum)) == 0) {
            m_data = { address, size };

            if (tf_tag) {
              m_tag = tf_tag;
            }

            return true;
        }
    }

    m_tag = 0;

    return false;
}


bool xmrig::WalletAddress::decode(const rapidjson::Value &address)
{
    return address.IsString() && decode(address.GetString(), address.GetStringLength());
}


const char *xmrig::WalletAddress::netName() const
{
    static const std::array<const char *, 3> names = { "mainnet", "testnet", "stagenet" };

    return names[net()];
}


const char *xmrig::WalletAddress::typeName() const
{
    static const std::array<const char *, 3> names = { "public", "integrated", "subaddress" };

    return names[type()];
}


rapidjson::Value xmrig::WalletAddress::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    return isValid() ? m_data.toJSON(doc) : Value(kNullType);
}


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::WalletAddress::toAPI(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    if (!isValid()) {
        return Value(kNullType);
    }

    auto &allocator = doc.GetAllocator();
    Value out(kObjectType);
    out.AddMember(StringRef(Coin::kField),  coin().toJSON(), allocator);
    out.AddMember("address",                m_data.toJSON(doc), allocator);
    out.AddMember("type",                   StringRef(typeName()), allocator);
    out.AddMember("net",                    StringRef(netName()), allocator);
    out.AddMember("rpc_port",               rpcPort(), allocator);
    out.AddMember("zmq_port",               zmqPort(), allocator);
    out.AddMember("tag",                    m_tag, allocator);
    out.AddMember("view_key",               Cvt::toHex(m_publicViewKey, kKeySize, doc), allocator);
    out.AddMember("spend_key",              Cvt::toHex(m_publicSpendKey, kKeySize, doc), allocator);
    out.AddMember("checksum",               Cvt::toHex(m_checksum, sizeof(m_checksum), doc), allocator);

    return out;
}
#endif


const xmrig::WalletAddress::TagInfo &xmrig::WalletAddress::tagInfo(uint64_t tag)
{
    static TagInfo dummy = { Coin::INVALID, MAINNET, PUBLIC, 0, 0 };
    static const std::map<uint64_t, TagInfo> tags = {
        { 0x12,     { Coin::MONERO,     MAINNET,    PUBLIC,         18081,  18082 } },
        { 0x13,     { Coin::MONERO,     MAINNET,    INTEGRATED,     18081,  18082 } },
        { 0x2a,     { Coin::MONERO,     MAINNET,    SUBADDRESS,     18081,  18082 } },

        { 0x35,     { Coin::MONERO,     TESTNET,    PUBLIC,         28081,  28082 } },
        { 0x36,     { Coin::MONERO,     TESTNET,    INTEGRATED,     28081,  28082 } },
        { 0x3f,     { Coin::MONERO,     TESTNET,    SUBADDRESS,     28081,  28082 } },

        { 0x18,     { Coin::MONERO,     STAGENET,   PUBLIC,         38081,  38082 } },
        { 0x19,     { Coin::MONERO,     STAGENET,   INTEGRATED,     38081,  38082 } },
        { 0x24,     { Coin::MONERO,     STAGENET,   SUBADDRESS,     38081,  38082 } },

        { 0x2bb39a, { Coin::SUMO,       MAINNET,    PUBLIC,         19734,  19735 } },
        { 0x29339a, { Coin::SUMO,       MAINNET,    INTEGRATED,     19734,  19735 } },
        { 0x8319a,  { Coin::SUMO,       MAINNET,    SUBADDRESS,     19734,  19735 } },

        { 0x37751a, { Coin::SUMO,       TESTNET,    PUBLIC,         29734,  29735 } },
        { 0x34f51a, { Coin::SUMO,       TESTNET,    INTEGRATED,     29734,  29735 } },
        { 0x1d351a, { Coin::SUMO,       TESTNET,    SUBADDRESS,     29734,  29735 } },

        { 0x2cca,   { Coin::ARQMA,      MAINNET,    PUBLIC,         19994,  19995 } },
        { 0x116bc7, { Coin::ARQMA,      MAINNET,    INTEGRATED,     19994,  19995 } },
        { 0x6847,   { Coin::ARQMA,      MAINNET,    SUBADDRESS,     19994,  19995 } },

        { 0x53ca,   { Coin::ARQMA,      TESTNET,    PUBLIC,         29994,  29995 } },
        { 0x504a,   { Coin::ARQMA,      TESTNET,    INTEGRATED,     29994,  29995 } },
        { 0x524a,   { Coin::ARQMA,      TESTNET,    SUBADDRESS,     29994,  29995 } },

        { 0x39ca,   { Coin::ARQMA,      STAGENET,   PUBLIC,         39994,  39995 } },
        { 0x1742ca, { Coin::ARQMA,      STAGENET,   INTEGRATED,     39994,  39995 } },
        { 0x1d84ca, { Coin::ARQMA,      STAGENET,   SUBADDRESS,     39994,  39995 } },

        { 0x1032,   { Coin::WOWNERO,    MAINNET,    PUBLIC,         34568,  34569 } },
        { 0x1a9a,   { Coin::WOWNERO,    MAINNET,    INTEGRATED,     34568,  34569 } },
        { 0x2fb0,   { Coin::WOWNERO,    MAINNET,    SUBADDRESS,     34568,  34569 } },

        { 0x5a,     { Coin::GRAFT,      MAINNET,    PUBLIC,         18981,  18982 } },
        { 0x5b,     { Coin::GRAFT,      MAINNET,    INTEGRATED,     18981,  18982 } },
        { 0x66,     { Coin::GRAFT,      MAINNET,    SUBADDRESS,     18981,  18982 } },

        { 0x54,     { Coin::GRAFT,      TESTNET,    PUBLIC,         28881,  28882 } },
        { 0x55,     { Coin::GRAFT,      TESTNET,    INTEGRATED,     28881,  28882 } },
        { 0x70,     { Coin::GRAFT,      TESTNET,    SUBADDRESS,     28881,  28882 } },

        { 0x424200,     { Coin::TOWNFORGE,     MAINNET,    PUBLIC,         18881,  18882 } },
        { 0x424201,     { Coin::TOWNFORGE,     MAINNET,    SUBADDRESS,     18881,  18882 } },

        { 0x424210,     { Coin::TOWNFORGE,     TESTNET,    PUBLIC,         28881,  28882 } },
        { 0x424211,     { Coin::TOWNFORGE,     TESTNET,    SUBADDRESS,     28881,  28882 } },

        { 0x424220,     { Coin::TOWNFORGE,     STAGENET,   PUBLIC,         38881,  38882 } },
        { 0x424221,     { Coin::TOWNFORGE,     STAGENET,   SUBADDRESS,     38881,  38882 } },

    };

    const auto it = tags.find(tag);

    return it == tags.end() ? dummy : it->second;
}
