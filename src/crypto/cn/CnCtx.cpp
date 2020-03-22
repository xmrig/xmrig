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

#include <limits>


#include "crypto/cn/CnCtx.h"
#include "base/crypto/Algorithm.h"
#include "crypto/cn/CryptoNight.h"
#include "crypto/common/portable/mm_malloc.h"
#include "crypto/common/VirtualMemory.h"


void xmrig::CnCtx::create(cryptonight_ctx **ctx, uint8_t *memory, size_t size, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        cryptonight_ctx *c = static_cast<cryptonight_ctx *>(_mm_malloc(sizeof(cryptonight_ctx), 4096));
        c->memory          = memory + (i * size);

        c->generated_code              = reinterpret_cast<cn_mainloop_fun_ms_abi>(VirtualMemory::allocateExecutableMemory(0x4000));
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
