/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include "crypto/common/Nonce.h"


namespace xmrig {

std::atomic<bool> Nonce::m_paused = {true};
std::atomic<uint64_t>  Nonce::m_sequence[Nonce::MAX] = { {1}, {1}, {1} };
std::atomic<uint64_t> Nonce::m_nonces[2] = { {0}, {0} };


} // namespace xmrig


bool xmrig::Nonce::next(uint8_t index, uint32_t *nonce, uint32_t reserveCount, uint64_t mask)
{
    mask &= 0x7FFFFFFFFFFFFFFFULL;
    if (reserveCount == 0 || mask < reserveCount - 1) {
        return false;
    }

    uint64_t counter = m_nonces[index].fetch_add(reserveCount, std::memory_order_relaxed);
    while (true) {
        if (mask < counter) {
            return false;
        }
        else if (mask - counter <= reserveCount - 1) {
            pause(true);
            if (mask - counter < reserveCount - 1) {
                return false;
            }
        }
        else if (0xFFFFFFFFUL - (uint32_t)counter < reserveCount - 1) {
            counter = m_nonces[index].fetch_add(reserveCount, std::memory_order_relaxed);
            continue;
        }
        *nonce = (nonce[0] & ~mask) | counter;
        if (mask > 0xFFFFFFFFULL) {
            nonce[1] = (nonce[1] & (~mask >> 32)) | (counter >> 32);
        }
        return true;
    }
}


void xmrig::Nonce::stop()
{
    pause(false);

    for (auto &i : m_sequence) {
        i = 0;
    }
}


void xmrig::Nonce::touch()
{
    for (auto &i : m_sequence) {
        i++;
    }
}
