/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
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


#include "crypto/randomx/randomx.h"
#include "crypto/rx/RxAlgo.h"


xmrig::Algorithm::Id xmrig::RxAlgo::apply(Algorithm::Id algorithm)
{
    switch (algorithm) {
    case Algorithm::RX_WOW:
        randomx_apply_config(RandomX_WowneroConfig);
        break;

    case Algorithm::RX_LOKI:
        randomx_apply_config(RandomX_LokiConfig);
        break;

    default:
        randomx_apply_config(RandomX_MoneroConfig);
        break;
    }

    return algorithm;
}


size_t xmrig::RxAlgo::l3(Algorithm::Id algorithm)
{
    switch (algorithm) {
    case Algorithm::RX_0:
        return RandomX_MoneroConfig.ScratchpadL3_Size;

    case Algorithm::RX_WOW:
        return RandomX_WowneroConfig.ScratchpadL3_Size;

    case Algorithm::RX_LOKI:
        return RandomX_LokiConfig.ScratchpadL3_Size;

    default:
        break;
    }

    return 0;
}
