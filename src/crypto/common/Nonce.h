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

#ifndef XMRIG_NONCE_H
#define XMRIG_NONCE_H


#include <atomic>


namespace xmrig {


class Nonce
{
public:
    enum Backend {
        CPU,
        OPENCL,
        CUDA,
        MAX
    };


    Nonce();

    static inline bool isOutdated(Backend backend, uint64_t sequence)   { return m_sequence[backend].load(std::memory_order_relaxed) != sequence; }
    static inline bool isPaused()                                       { return m_paused.load(std::memory_order_relaxed); }
    static inline uint64_t sequence(Backend backend)                    { return m_sequence[backend].load(std::memory_order_relaxed); }
    static inline void pause(bool paused)                               { m_paused = paused; }
    static inline void stop(Backend backend)                            { m_sequence[backend] = 0; }
    static inline void touch(Backend backend)                           { m_sequence[backend]++; }

    static uint32_t next(uint8_t index, uint32_t nonce, uint32_t reserveCount, bool nicehash);
    static void reset(uint8_t index);
    static void stop();
    static void touch();

private:
    static std::atomic<bool> m_paused;
    static std::atomic<uint64_t> m_sequence[MAX];
    static uint32_t m_nonces[2];
};


} // namespace xmrig


#endif /* XMRIG_NONCE_H */
