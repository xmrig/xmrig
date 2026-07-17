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

#ifndef XMRIG_RXTARI_H
#define XMRIG_RXTARI_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculates a RandomX hash for a Tari (76-byte) mining blob.
 *
 * Tari uses a different blob format than standard Monero RandomX:
 *   - Bytes 0-2:    flags/version (3 bytes: major, minor, timestamp)
 *   - Bytes 3-34:   mining_hash (32 bytes)
 *   - Bytes 35-42:  nonce (8 bytes, little-endian u64)
 *   - Bytes 43-75:  pow_data (33 bytes: 1 byte algo + 32 bytes data)
 *
 * The Tari node hashes the full 76-byte blob directly with RandomX.
 * This function does the same — no repacking into Monero format is needed or desired.
 *
 * @param machine   RandomX VM (initialized with Tari config)
 * @param input     Pointer to the 76-byte Tari mining blob
 * @param inputSize Size of the Tari blob (should be 76)
 * @param output    Output buffer for the 32-byte hash (RANDOMX_HASH_SIZE)
 */
void tari_randomx_calculate_hash(void* machine, const void* input, size_t inputSize, void* output);

#ifdef __cplusplus
}
#endif

#endif /* XMRIG_RXTARI_H */
