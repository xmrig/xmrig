/* XMRig
 * Copyright (c) 2018-2026 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2026 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "crypto/rx/RxTari.h"
#include "crypto/randomx/randomx.h"
#include <cstddef>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tari RandomXT blob format (76 bytes, as produced by Tari node):
 *   Bytes 0-2:    flags/version (3 bytes: major, minor, timestamp)
 *   Bytes 3-34:   mining_hash (32 bytes)
 *   Bytes 35-42:  nonce (8 bytes, little-endian u64)
 *   Bytes 43-75:  pow_data (33 bytes: 1 byte algo + 32 bytes data)
 *
 * The Tari node hashes the full 76-byte blob directly with RandomX.
 * We do the same — no repacking is needed or desired.
 */

#define TARI_BLOB_SIZE 76

void tari_randomx_calculate_hash(void* machine, const void* input, size_t inputSize, void* output)
{
    if (!machine || !input || !output) {
        return;
    }

    // Hash the full 76-byte Tari blob directly.
    // The Tari node does this — no repacking into Monero format.
    randomx_calculate_hash(static_cast<randomx_vm*>(machine), input, TARI_BLOB_SIZE, output);
}

#ifdef __cplusplus
}
#endif
