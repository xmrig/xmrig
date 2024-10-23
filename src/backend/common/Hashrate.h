/* XMRig
 * Copyright (c) 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_HASHRATE_H
#define XMRIG_HASHRATE_H


#include <cmath>
#include <cstddef>
#include <cstdint>


#include "3rdparty/rapidjson/fwd.h"
#include "base/tools/Object.h"


namespace xmrig {


class Hashrate
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Hashrate)

    enum Intervals : size_t {
        ShortInterval  = 10000,
        MediumInterval = 60000,
        LargeInterval  = 900000
    };

    Hashrate(size_t threads);
    ~Hashrate();

    inline std::pair<bool, double> calc(size_t ms) const                    { return hashrate(0U, ms); }
    inline std::pair<bool, double> calc(size_t threadId, size_t ms) const   { return hashrate(threadId + 1, ms); }
    inline size_t threads() const                                           { return m_threads > 0U ? m_threads - 1U : 0U; }
    inline void add(size_t threadId, uint64_t count, uint64_t timestamp)    { addData(threadId + 1U, count, timestamp); }
    inline void add(uint64_t count, uint64_t timestamp)                     { addData(0U, count, timestamp); }

    double average() const;

    static const char *format(std::pair<bool, double> h, char *buf, size_t size);
    static rapidjson::Value normalize(std::pair<bool, double> d);

#   ifdef XMRIG_FEATURE_API
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    rapidjson::Value toJSON(size_t threadId, rapidjson::Document &doc) const;
#   endif

private:
    std::pair<bool, double> hashrate(size_t index, size_t ms) const;
    void addData(size_t index, uint64_t count, uint64_t timestamp);

    constexpr static size_t kBucketSize = 2 << 11;
    constexpr static size_t kBucketMask = kBucketSize - 1;

    size_t m_threads;
    uint32_t* m_top;
    uint64_t** m_counts;
    uint64_t** m_timestamps;

    uint64_t m_earliestTimestamp;
    uint64_t m_totalCount;
};


} // namespace xmrig


#endif /* XMRIG_HASHRATE_H */
