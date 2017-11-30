/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
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


#include <memory.h>


#include "crypto/CryptoNight.h"
#include "Mem.h"
#include "Options.h"


bool Mem::m_doubleHash   = false;
int Mem::m_algo          = 0;
int Mem::m_flags         = 0;
int Mem::m_threads       = 0;
size_t Mem::m_memorySize = 0;
uint8_t *Mem::m_memory   = nullptr;
int64_t Mem::m_doubleHashThreadMask = -1L;

cryptonight_ctx *Mem::create(int threadId)
{
    size_t scratchPadSize = m_algo == Options::ALGO_CRYPTONIGHT ? MEMORY : MEMORY_LITE;

    size_t offset = 0;
    for (int i=0; i < threadId; i++) {
        offset += sizeof(cryptonight_ctx);
        offset += isDoubleHash(i) ? scratchPadSize*2 : scratchPadSize;
    }

    auto* ctx = reinterpret_cast<cryptonight_ctx *>(&m_memory[offset]);

    size_t memOffset = offset+sizeof(cryptonight_ctx);

    ctx->memory = &m_memory[memOffset];

    return ctx;
}
