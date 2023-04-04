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

#ifndef XMRIG_SIGNATURES_H
#define XMRIG_SIGNATURES_H


#include <cstdint>


namespace xmrig {


void generate_signature(const uint8_t* prefix_hash, const uint8_t* pub, const uint8_t* sec, uint8_t* sig);
bool check_signature(const uint8_t* prefix_hash, const uint8_t* pub, const uint8_t* sig);

bool generate_key_derivation(const uint8_t* key1, const uint8_t* key2, uint8_t* derivation, uint8_t* view_tag);
void derive_secret_key(const uint8_t* derivation, size_t output_index, const uint8_t* base, uint8_t* derived_key);
bool derive_public_key(const uint8_t* derivation, size_t output_index, const uint8_t* base, uint8_t* derived_key);

void derive_view_secret_key(const uint8_t* spend_secret_key, uint8_t* view_secret_key);

void generate_keys(uint8_t* pub, uint8_t* sec);
bool secret_key_to_public_key(const uint8_t* sec, uint8_t* pub);

} /* namespace xmrig */


#endif /* XMRIG_SIGNATURES_H */
