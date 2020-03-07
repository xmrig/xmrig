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
#include "crypto/defyx/defyx.h"
#include "crypto/rx/RxAlgo.h"


xmrig::Algorithm::Id xmrig::RxAlgo::apply(Algorithm::Id algorithm)
{
    randomx_apply_config(*base(algorithm));

    return algorithm;
}


const RandomX_ConfigurationBase *xmrig::RxAlgo::base(Algorithm::Id algorithm)
{
    switch (algorithm) {
    case Algorithm::RX_WOW:
        return &RandomX_WowneroConfig;

    case Algorithm::RX_LOKI:
        return &RandomX_LokiConfig;

    case Algorithm::RX_ARQ:
        return &RandomX_ArqmaConfig;

    case Algorithm::DEFYX:
        return &RandomX_ScalaConfig;
        break;

    case Algorithm::RX_SFX:
        return &RandomX_SafexConfig;

    case Algorithm::RX_KEVA:
        return &RandomX_KevaConfig;

    default:
        break;
    }

    return &RandomX_MoneroConfig;
}


uint32_t xmrig::RxAlgo::version(Algorithm::Id algorithm)
{
    return algorithm == Algorithm::RX_WOW ? 103 : 104;
}


uint32_t xmrig::RxAlgo::programCount(Algorithm::Id algorithm)
{
    return base(algorithm)->ProgramCount;
}


uint32_t xmrig::RxAlgo::programIterations(Algorithm::Id algorithm)
{
    return base(algorithm)->ProgramIterations;
}


uint32_t xmrig::RxAlgo::programSize(Algorithm::Id algorithm)
{
    return base(algorithm)->ProgramSize;
}
