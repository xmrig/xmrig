/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef __CRYPTONIGHT_MONERO_H__
#define __CRYPTONIGHT_MONERO_H__


// VARIANT ALTERATIONS
#ifndef XMRIG_ARM
#   define VARIANT1_INIT(part) \
    uint64_t tweak1_2_##part = 0; \
    if (IS_MONERO) { \
        tweak1_2_##part = (*reinterpret_cast<const uint64_t*>(input + 35 + part * size) ^ \
                          *(reinterpret_cast<const uint64_t*>(ctx[part]->state) + 24)); \
    }
#else
#   define VARIANT1_INIT(part) \
    uint64_t tweak1_2_##part = 0; \
    if (IS_MONERO) { \
        memcpy(&tweak1_2_##part, input + 35 + part * size, sizeof tweak1_2_##part); \
        tweak1_2_##part ^= *(reinterpret_cast<const uint64_t*>(ctx[part]->state) + 24); \
    }
#endif

#define VARIANT1_1(p) \
    if (IS_MONERO) { \
        const uint8_t tmp = reinterpret_cast<const uint8_t*>(p)[11]; \
        static const uint32_t table = 0x75310; \
        const uint8_t index = (((tmp >> 3) & 6) | (tmp & 1)) << 1; \
        ((uint8_t*)(p))[11] = tmp ^ ((table >> index) & 0x30); \
    }

#define VARIANT1_2(p, part) \
    if (IS_MONERO) { \
        (p) ^= tweak1_2_##part; \
    }


#endif /* __CRYPTONIGHT_MONERO_H__ */
