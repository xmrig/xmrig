/* XMRig
 * Copyright 2012-2013 The Cryptonote developers
 * Copyright 2014-2021 The Monero Project
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/crypto/keccak.h"
#include "base/tools/cryptonote/BlobReader.h"
#include "base/tools/cryptonote/WalletAddress.h"
#include "base/tools/cryptonote/umul128.h"
#include "base/tools/Buffer.h"
#include <array>


namespace xmrig {


bool WalletAddress::Decode(const String& address)
{
    static constexpr std::array<int, 9> block_sizes{ 0, 2, 3, 5, 6, 7, 9, 10, 11 };
    static constexpr char alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    constexpr size_t alphabet_size = sizeof(alphabet) - 1;

    int8_t reverse_alphabet[256];
    memset(reverse_alphabet, -1, sizeof(reverse_alphabet));

    for (size_t i = 0; i < alphabet_size; ++i) {
        reverse_alphabet[static_cast<int>(alphabet[i])] = i;
    }

    const int len = static_cast<int>(address.size());
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

    Buffer data;
    data.reserve(static_cast<size_t>(num_full_blocks) * sizeof(uint64_t) + last_block_size_index);

    const char* address_data = address.data();

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

        uint8_t* p = reinterpret_cast<uint8_t*>(&num);
        for (int j = ((i < num_full_blocks) ? sizeof(num) : last_block_size_index) - 1; j >= 0; --j) {
            data.emplace_back(p[j]);
        }
    }

    CBlobReader ar(data.data(), data.size());

    ar(tag);
    ar(public_spend_key);
    ar(public_view_key);
    ar(checksum);

    uint8_t md[200];
    keccak(data.data(), data.size() - sizeof(checksum), md);

    if (memcmp(checksum, md, sizeof(checksum)) != 0) {
        return false;
    }

    return true;
}


} /* namespace xmrig */
