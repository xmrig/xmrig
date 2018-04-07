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


int Mem::m_algo          = 0;
int Mem::m_flags         = 0;
size_t Mem::m_hashFactor = 1;
size_t Mem::m_threads    = 0;
size_t Mem::m_memorySize = 0;
alignas(16) uint8_t *Mem::m_memory = nullptr;
Mem::ThreadBitSet Mem::m_multiHashThreadMask = Mem::ThreadBitSet(-1L);

cryptonight_ctx *Mem::create(int threadId)
{
    size_t scratchPadSize;

    switch (m_algo)
    {
        case Options::ALGO_CRYPTONIGHT_LITE:
            scratchPadSize = MEMORY_LITE;
            break;
        case Options::ALGO_CRYPTONIGHT_HEAVY:
            scratchPadSize = MEMORY_HEAVY;
            break;
        case Options::ALGO_CRYPTONIGHT:
        default:
            scratchPadSize = MEMORY;
            break;
    }

    size_t offset = 0;
    for (int i=0; i < threadId; i++) {
        offset += sizeof(cryptonight_ctx);
        offset += scratchPadSize * getThreadHashFactor(i);
    }

    auto* ctx = reinterpret_cast<cryptonight_ctx *>(&m_memory[offset]);

    size_t memOffset = offset+sizeof(cryptonight_ctx);

    ctx->memory = &m_memory[memOffset];

    return ctx;
}
