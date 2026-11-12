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

#ifndef XMRIG_NONCE_H
#define XMRIG_NONCE_H


#include <atomic>
#include <cstdint>
#include <mutex>


namespace xmrig {


// Best-hash tracking: keeps the best (lowest) raw RandomX/CN hash value per job and per
// session, plus the session top-10 record list. A lower raw value means a more difficult
// hash; its difficulty is Job::toDiff(value), the same conversion JobResult::actualDiff()
// uses for found shares.
constexpr size_t kBestHashListSize = 10;

class Nonce
{
public:
    enum Backend : uint32_t {
        CPU,
        OPENCL,
        CUDA,
        MAX
    };


    static inline bool isOutdated(Backend backend, uint64_t sequence)   { return m_sequence[backend].load(std::memory_order_relaxed) != sequence; }
    static inline bool isPaused()                                       { return m_paused.load(std::memory_order_relaxed); }
    static inline uint64_t sequence(Backend backend)                    { return m_sequence[backend].load(std::memory_order_relaxed); }
    static inline void pause(bool paused)                               { m_paused = paused; }
    static inline void reset(uint8_t index)                             { m_nonces[index] = 0; }
    static inline void stop(Backend backend)                            { m_sequence[backend] = 0; }
    static inline void touch(Backend backend)                           { m_sequence[backend]++; }

    static inline uint64_t bestHashValue()                              { return m_bestHashValue.load(std::memory_order_relaxed); }
    static inline uint64_t sessionBestHashValue()                       { return m_sessionBestHashValue.load(std::memory_order_relaxed); }

    // Called for every computed hash value. Fast path: one relaxed load per tracker;
    // the record list is touched only when a hash beats the worst entry currently in it.
    static inline void submitHashValue(uint64_t value)
    {
        uint64_t best = m_bestHashValue.load(std::memory_order_relaxed);
        if (best == 0 || value < best) {
            while (best == 0 || value < best) {
                if (m_bestHashValue.compare_exchange_weak(best, value, std::memory_order_relaxed)) {
                    break;
                }
            }
        }

        // Checked unconditionally: a hash that misses the stale per-job best seen here
        // (e.g. across a job change) can still be a new session best.
        best = m_sessionBestHashValue.load(std::memory_order_relaxed);
        while (best == 0 || value < best) {
            if (m_sessionBestHashValue.compare_exchange_weak(best, value, std::memory_order_relaxed)) {
                break;
            }
        }

        // Session top-10 list: m_recordWorst is the largest value currently in the list
        // (0 while the list is not full). Values that don't beat it can't belong there.
        const uint64_t worst = m_recordWorst.load(std::memory_order_relaxed);
        if (worst == 0 || value < worst) {
            submitRecord(value);
        }
    }

    // Session top-10 record list, best (lowest value = highest diff) first. Copies a
    // consistent snapshot under the same lock the updater holds; unused slots are 0.
    static inline void readBestHashList(uint64_t *out, size_t &count)
    {
        std::lock_guard<std::mutex> lock(m_recordsMutex);

        count = m_recordCount;
        for (size_t i = 0; i < count; ++i) {
            out[i] = m_recordList[i];
        }
    }

    static bool next(uint8_t index, uint32_t *nonce, uint32_t reserveCount, uint64_t mask);
    static void stop();
    static void touch();

private:
    // Insert a value into the session top-10 record list. Called only when `value`
    // beats the worst entry currently in the list (or the list is not full yet).
    // m_recordsMutex serializes inserts; readers take the same lock for a snapshot.
    static void submitRecord(uint64_t value);

    static std::atomic<bool> m_paused;
    static std::atomic<uint64_t> m_sequence[MAX];
    static std::atomic<uint64_t> m_nonces[2];
    static std::atomic<uint64_t> m_bestHashValue;
    static std::atomic<uint64_t> m_sessionBestHashValue;
    static std::atomic<uint64_t> m_recordWorst;
    static std::mutex m_recordsMutex;
    static uint64_t m_recordList[kBestHashListSize];
    static size_t m_recordCount;
};


} // namespace xmrig


#endif /* XMRIG_NONCE_H */
