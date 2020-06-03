/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_KP_HASH_H
#define XMRIG_KP_HASH_H


#include <stdint.h>


namespace xmrig
{


class KPCache;


class KPHash
{
public:
    static constexpr uint32_t EPOCH_LENGTH  = 7500;
    static constexpr uint32_t PERIOD_LENGTH = 3;
    static constexpr int CNT_CACHE = 11;
    static constexpr int CNT_MATH = 18;
    static constexpr uint32_t REGS = 32;
    static constexpr uint32_t LANES = 16;

    static void calculate(const KPCache& light_cache, uint32_t block_height, const uint8_t (&header_hash)[32], uint64_t nonce, uint32_t (&output)[8], uint32_t (&mix_hash)[8]);
};


} /* namespace xmrig */


#endif /* XMRIG_KP_HASH_H */
