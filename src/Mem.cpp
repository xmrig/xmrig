/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include <limits>


#include "crypto/cn/CryptoNight.h"
#include "crypto/common/portable/mm_malloc.h"
#include "crypto/common/VirtualMemory.h"
#include "Mem.h"


bool Mem::m_enabled = true;
int Mem::m_flags    = 0;


MemInfo Mem::create(cryptonight_ctx **ctx, const xmrig::Algorithm &algorithm, size_t count)
{
    using namespace xmrig;

    constexpr CnAlgo<Algorithm::CN_0> props;

    MemInfo info;
    info.size = props.memory(algorithm.id()) * count;

    constexpr const size_t align_size = 2 * 1024 * 1024;
    info.size  = ((info.size + align_size - 1) / align_size) * align_size;
    info.pages = info.size / align_size;

    allocate(info, m_enabled);

    for (size_t i = 0; i < count; ++i) {
        cryptonight_ctx *c = static_cast<cryptonight_ctx *>(_mm_malloc(sizeof(cryptonight_ctx), 4096));
        c->memory          = info.memory + (i * props.memory(algorithm.id()));

        c->generated_code              = reinterpret_cast<cn_mainloop_fun_ms_abi>(VirtualMemory::allocateExecutableMemory(0x4000));
        c->generated_code_data.algo    = Algorithm::INVALID;
        c->generated_code_data.height  = std::numeric_limits<uint64_t>::max();

        ctx[i] = c;
    }

    return info;
}


void Mem::release(cryptonight_ctx **ctx, size_t count, MemInfo &info)
{
    release(info);

    for (size_t i = 0; i < count; ++i) {
        _mm_free(ctx[i]);
    }
}

