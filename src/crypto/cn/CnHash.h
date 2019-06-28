/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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

#ifndef XMRIG_CN_HASH_H
#define XMRIG_CN_HASH_H


#include <stddef.h>
#include <stdint.h>


#include "common/xmrig.h"
#include "crypto/cn/CnAlgo.h"
#include "crypto/common/Assembly.h"


struct cryptonight_ctx;


namespace xmrig
{

typedef void (*cn_hash_fun)(const uint8_t *input, size_t size, uint8_t *output, cryptonight_ctx **ctx, uint64_t height);
typedef void (*cn_mainloop_fun)(cryptonight_ctx **ctx);


class CnHash
{
public:
    CnHash();

    cn_hash_fun fn(const Algorithm &algorithm, AlgoVariant av, Assembly::Id assembly) const;

private:
    cn_hash_fun m_map[Algorithm::MAX][AV_MAX][Assembly::MAX] = {};
};


} /* namespace xmrig */


#endif /* XMRIG_CN_HASH_H */
