/* XMRig
 * Copyright (c) 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include <limits>


#include "crypto/cn/CnCtx.h"
#include "base/crypto/Algorithm.h"
#include "crypto/cn/CryptoNight.h"
#include "crypto/common/portable/mm_malloc.h"
#include "crypto/common/VirtualMemory.h"


void xmrig::CnCtx::create(cryptonight_ctx **ctx, uint8_t *memory, size_t size, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        auto *c     = static_cast<cryptonight_ctx *>(_mm_malloc(sizeof(cryptonight_ctx), 4096));
        c->memory   = memory + (i * size);

        c->generated_code              = reinterpret_cast<cn_mainloop_fun_ms_abi>(VirtualMemory::allocateExecutableMemory(0x4000, false));
        c->generated_code_data.algo    = Algorithm::INVALID;
        c->generated_code_data.height  = std::numeric_limits<uint64_t>::max();

        ctx[i] = c;
    }
}


void xmrig::CnCtx::release(cryptonight_ctx **ctx, size_t count)
{
    if (ctx[0] == nullptr) {
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        _mm_free(ctx[i]);
    }
}
