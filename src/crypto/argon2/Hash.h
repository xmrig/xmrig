/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_ARGON2_HASH_H
#define XMRIG_ARGON2_HASH_H


#include "3rdparty/argon2.h"
#include "base/crypto/Algorithm.h"
#include "crypto/cn/CryptoNight.h"


namespace xmrig { namespace argon2 {


template<Algorithm::Id ALGO>
inline void single_hash(const uint8_t *__restrict__ input, size_t size, uint8_t *__restrict__ output, cryptonight_ctx **__restrict__ ctx, uint64_t)
{
    if (ALGO == Algorithm::AR2_CHUKWA) {
        argon2id_hash_raw_ex(3, 512, 1, input, size, input, 16, output, 32, ctx[0]->memory);
    }
    else if (ALGO == Algorithm::AR2_WRKZ) {
        argon2id_hash_raw_ex(4, 256, 1, input, size, input, 16, output, 32, ctx[0]->memory);
    }
}


}} // namespace xmrig::argon2


#endif /* XMRIG_ARGON2_HASH_H */
