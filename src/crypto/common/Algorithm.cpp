/* XMRig
 * Copyright (c) 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "crypto/common/Algorithm.h"


#include <set>


xmrig_cuda::Algorithm::Id xmrig_cuda::Algorithm::parse(uint32_t id)
{
    static const std::set<uint32_t> ids = {
        CN_0, CN_1, CN_2, CN_FAST, CN_HALF, CN_XAO, CN_RTO, CN_RWZ, CN_ZLS, CN_DOUBLE, CN_CCX,
#       ifdef XMRIG_ALGO_CN_R
        CN_R,
#       endif
#       ifdef XMRIG_ALGO_CN_LITE
        CN_LITE_0, CN_LITE_1,
#       endif
#       ifdef XMRIG_ALGO_CN_HEAVY
        CN_HEAVY_0, CN_HEAVY_TUBE, CN_HEAVY_XHV,
#       endif
#       ifdef XMRIG_ALGO_CN_PICO
        CN_PICO_0, CN_PICO_TLO,
#       endif
#       ifdef XMRIG_ALGO_CN_FEMTO
        CN_UPX2,
#       endif
#       ifdef XMRIG_ALGO_RANDOMX
        RX_0, RX_WOW, RX_ARQ, RX_GRAFT, RX_SFX, RX_KEVA,
#       endif
#       ifdef XMRIG_ALGO_ARGON2
        AR2_CHUKWA, AR2_CHUKWA_V2, AR2_WRKZ,
#       endif
#       ifdef XMRIG_ALGO_ASTROBWT
        ASTROBWT_DERO,
#       endif
#       ifdef XMRIG_ALGO_KAWPOW
        KAWPOW_RVN,
#       endif
    };

    return ids.count(id) ? static_cast<Id>(id) : INVALID;
}
