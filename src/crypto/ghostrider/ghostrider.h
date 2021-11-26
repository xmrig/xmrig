/* XMRig
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

#ifndef XMRIG_GR_HASH_H
#define XMRIG_GR_HASH_H


#include <cstddef>
#include <cstdint>
#include <vector>


struct cryptonight_ctx;


namespace xmrig
{


namespace ghostrider
{


struct HelperThread;

void benchmark();
HelperThread* create_helper_thread(int64_t cpu_index, const std::vector<int64_t>& affinities);
void destroy_helper_thread(HelperThread* t);
void hash_octa(const uint8_t* data, size_t size, uint8_t* output, cryptonight_ctx** ctx, HelperThread* helper);


} // namespace ghostrider


} // namespace xmrig

#endif // XMRIG_GR_HASH_H