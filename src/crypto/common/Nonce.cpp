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


#include <algorithm>


namespace xmrig {

std::atomic<bool> Nonce::m_paused = {true};
std::atomic<uint64_t>  Nonce::m_sequence[Nonce::MAX] = { {1}, {1}, {1} };
std::atomic<uint64_t> Nonce::m_nonces[2] = { {0}, {0} };
std::atomic<uint64_t> Nonce::m_bestHashValue = {0};
std::atomic<uint64_t> Nonce::m_sessionBestHashValue = {0};
std::atomic<uint64_t> Nonce::m_recordWorst = {0};

std::mutex Nonce::m_recordsMutex;
uint64_t Nonce::m_recordList[kBestHashListSize] = { 0 };
size_t Nonce::m_recordCount = 0;


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

    // New job: restart the best-actual-hash-difficulty tracking.
    m_bestHashValue = 0;
}


// Insert a value into the session top-10 record list. The caller has already checked
// that `value` beats m_recordWorst, but the check must be repeated under the lock:
// several workers can pass the lock-free gate at the same time on the same record.
void xmrig::Nonce::submitRecord(uint64_t value)
{
    std::lock_guard<std::mutex> lock(m_recordsMutex);

    if (m_recordCount > 0 && value >= m_recordList[m_recordCount - 1]) {
        return;
    }

    if (m_recordCount < kBestHashListSize) {
        // List not full yet: append, then insertion-sort the new entry into place.
        m_recordList[m_recordCount++] = value;
        std::sort(m_recordList, m_recordList + m_recordCount);

        if (m_recordCount == kBestHashListSize) {
            m_recordWorst.store(m_recordList[kBestHashListSize - 1], std::memory_order_relaxed);
        }

        return;
    }

    // List full: drop the worst entry, append the new one, re-sort, publish the new worst.
    m_recordList[kBestHashListSize - 1] = value;
    std::sort(m_recordList, m_recordList + kBestHashListSize);
    m_recordWorst.store(m_recordList[kBestHashListSize - 1], std::memory_order_relaxed);
}
