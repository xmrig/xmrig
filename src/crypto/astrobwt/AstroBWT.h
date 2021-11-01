/* XMRig
 * Copyright (c) 2018      Lee Clagett              <https://github.com/vtnerd>
 * Copyright (c) 2018-2019 tevador                  <tevador@gmail.com>
 * Copyright (c) 2000      Transmeta Corporation    <https://github.com/intel/msr-tools>
 * Copyright (c) 2004-2008 H. Peter Anvin           <https://github.com/intel/msr-tools>
 * Copyright (c) 2018-2021 SChernykh                <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig                    <https://github.com/xmrig>, <support@xmrig.com>
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

#include "base/crypto/Algorithm.h"


struct cryptonight_ctx;


namespace xmrig {


namespace astrobwt {

bool astrobwt_dero(const void* input_data, uint32_t input_size, void* scratchpad, uint8_t* output_hash, int stage2_max_size, bool avx2);
void init();

template<Algorithm::Id ALGO>
void single_hash(const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx** ctx, uint64_t);

template<>
void single_hash<Algorithm::ASTROBWT_DERO>(const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx** ctx, uint64_t);


}} // namespace xmrig::astrobwt
