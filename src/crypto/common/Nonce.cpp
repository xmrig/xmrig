/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include <mutex>


#include "crypto/common/Nonce.h"


namespace xmrig {


std::atomic<bool> Nonce::m_paused;
std::atomic<uint64_t> Nonce::m_sequence[Nonce::MAX];
uint32_t Nonce::m_nonces[2] = { 0, 0 };


static std::mutex mutex;
static Nonce nonce;


} // namespace xmrig


xmrig::Nonce::Nonce()
{
    m_paused = true;

    for (int i = 0; i < MAX; ++i) {
        m_sequence[i] = 1;
    }
}


uint32_t xmrig::Nonce::next(uint8_t index, uint32_t nonce, uint32_t reserveCount, bool nicehash)
{
    uint32_t next;

    std::lock_guard<std::mutex> lock(mutex);

    if (nicehash) {
        next = (nonce & 0xFF000000) | m_nonces[index];
    }
    else {
        next = m_nonces[index];
    }

    m_nonces[index] += reserveCount;

    return next;
}


void xmrig::Nonce::reset(uint8_t index)
{
    std::lock_guard<std::mutex> lock(mutex);

    m_nonces[index] = 0;
}


void xmrig::Nonce::stop()
{
    pause(false);

    for (int i = 0; i < MAX; ++i) {
        m_sequence[i] = 0;
    }
}


void xmrig::Nonce::touch()
{
    for (int i = 0; i < MAX; ++i) {
        m_sequence[i]++;
    }
}
