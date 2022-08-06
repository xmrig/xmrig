/* XMRig
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

#include "base/tools/Alignment.h"
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

        if (mask - counter <= reserveCount - 1) {
            pause(true);
            if (mask - counter < reserveCount - 1) {
                return false;
            }
        }
        else if (0xFFFFFFFFUL - (uint32_t)counter < reserveCount - 1) {
            counter = m_nonces[index].fetch_add(reserveCount, std::memory_order_relaxed);
            continue;
        }

        writeUnaligned(nonce, static_cast<uint32_t>((readUnaligned(nonce) & ~mask) | counter));

        if (mask > 0xFFFFFFFFULL) {
            writeUnaligned(nonce + 1, static_cast<uint32_t>((readUnaligned(nonce + 1) & (~mask >> 32)) | (counter >> 32)));
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
