/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
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

#ifndef XMRIG_CRYPTONIGHT_H
#define XMRIG_CRYPTONIGHT_H


#include <stddef.h>
#include <stdint.h>

#if defined _MSC_VER || defined XMRIG_ARM
#   define ABI_ATTRIBUTE
#else
#   define ABI_ATTRIBUTE __attribute__((ms_abi))
#endif


struct cryptonight_ctx;
typedef void(*cn_mainloop_fun_ms_abi)(cryptonight_ctx**) ABI_ATTRIBUTE;


struct cryptonight_r_data {
    int algo;
    uint64_t height;

    bool match(const int a, const uint64_t h) const { return (a == algo) && (h == height); }
};


struct cryptonight_ctx {
    alignas(16) uint8_t state[224];
    alignas(16) uint8_t *memory;

    uint8_t unused[40];
    const uint32_t *saes_table;

    cn_mainloop_fun_ms_abi generated_code;
    cryptonight_r_data generated_code_data;
};


#endif /* XMRIG_CRYPTONIGHT_H */
