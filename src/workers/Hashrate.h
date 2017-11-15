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

#ifndef __HASHRATE_H__
#define __HASHRATE_H__


#include <stdint.h>
#include <uv.h>


class Hashrate
{
public:
    enum Intervals {
        ShortInterval  = 2500,
        MediumInterval = 60000,
        LargeInterval  = 900000
    };

    Hashrate(int threads);
    double calc(size_t ms) const;
    double calc(size_t threadId, size_t ms) const;
    void add(size_t threadId, uint64_t count, uint64_t timestamp);
    void print();
    void stop();
    void updateHighest();

    inline double highest() const { return m_highest; }
    inline int threads() const    { return m_threads; }

private:
    static void onReport(uv_timer_t *handle);

    constexpr static size_t kBucketSize = 2 << 11;
    constexpr static size_t kBucketMask = kBucketSize - 1;

    double m_highest;
    int m_threads;
    uint32_t* m_top;
    uint64_t** m_counts;
    uint64_t** m_timestamps;
    uv_timer_t m_timer;
};


#endif /* __HASHRATE_H__ */
