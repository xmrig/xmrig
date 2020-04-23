/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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

#ifndef XMRIG_CN_HASH_H
#define XMRIG_CN_HASH_H


#include <cstddef>
#include <cstdint>


#include "crypto/cn/CnAlgo.h"
#include "crypto/common/Assembly.h"


struct cryptonight_ctx;


namespace xmrig
{

using cn_hash_fun     = void (*)(const uint8_t *, size_t, uint8_t *, cryptonight_ctx **, uint64_t);
using cn_mainloop_fun = void (*)(cryptonight_ctx **);


class CnHash
{
public:
    enum AlgoVariant {
        AV_AUTO,        // --av=0 Automatic mode.
        AV_SINGLE,      // --av=1  Single hash mode
        AV_DOUBLE,      // --av=2  Double hash mode
        AV_SINGLE_SOFT, // --av=3  Single hash mode (Software AES)
        AV_DOUBLE_SOFT, // --av=4  Double hash mode (Software AES)
        AV_TRIPLE,      // --av=5  Triple hash mode
        AV_QUAD,        // --av=6  Quard hash mode
        AV_PENTA,       // --av=7  Penta hash mode
        AV_TRIPLE_SOFT, // --av=8  Triple hash mode (Software AES)
        AV_QUAD_SOFT,   // --av=9  Quard hash mode  (Software AES)
        AV_PENTA_SOFT,  // --av=10 Penta hash mode  (Software AES)
        AV_MAX
    };

    CnHash();

    static cn_hash_fun fn(const Algorithm &algorithm, AlgoVariant av, Assembly::Id assembly);

private:
    cn_hash_fun m_map[Algorithm::MAX][AV_MAX][Assembly::MAX] = {};
};


} /* namespace xmrig */


#endif /* XMRIG_CN_HASH_H */
