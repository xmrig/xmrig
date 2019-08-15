/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
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


#ifndef XMRIG_ARGON2_H
#define XMRIG_ARGON2_H


#include "3rdparty/argon2.h"
#include "crypto/common/Algorithm.h"


struct cryptonight_ctx;


namespace xmrig {


template<Algorithm::Id ALGO>
inline void argon2_single_hash(const uint8_t *__restrict__ input, size_t size, uint8_t *__restrict__ output, cryptonight_ctx **__restrict__, uint64_t)
{
//    static bool argon_optimization_selected = false;

//    if (!argon_optimization_selected) {
//        argon2_select_impl(stdout, nullptr);

//        argon_optimization_selected = true;
//    }

    uint8_t salt[16];

    memcpy(salt, input, sizeof(salt));

    if (ALGO == Algorithm::AR2_CHUKWA) {
        argon2id_hash_raw(3, 512, 1, input, size, salt, 16, output, 32);
    }
    else if (ALGO == Algorithm::AR2_WRKZ) {
        argon2id_hash_raw(4, 256, 1, input, size, salt, 16, output, 32);
    }
}


} // namespace xmrig


#endif /* XMRIG_ARGON2_H */
