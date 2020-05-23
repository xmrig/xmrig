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

#ifndef XMRIG_HASHRATE_H
#define XMRIG_HASHRATE_H


#include <cstddef>
#include <cstdint>


#include "3rdparty/rapidjson/fwd.h"
#include "base/tools/Object.h"


namespace xmrig {


class Hashrate
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Hashrate)

    enum Intervals {
        ShortInterval  = 10000,
        MediumInterval = 60000,
        LargeInterval  = 900000
    };

    Hashrate(size_t threads);
    ~Hashrate();
    double calc(size_t ms) const;
    double calc(size_t threadId, size_t ms) const;
    void add(size_t threadId, uint64_t count, uint64_t timestamp);

    inline size_t threads() const { return m_threads; }

    static const char *format(double h, char *buf, size_t size);
    static rapidjson::Value normalize(double d);

#   ifdef XMRIG_FEATURE_API
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    rapidjson::Value toJSON(size_t threadId, rapidjson::Document &doc) const;
#   endif

private:
    constexpr static size_t kBucketSize = 2 << 11;
    constexpr static size_t kBucketMask = kBucketSize - 1;

    size_t m_threads;
    uint32_t* m_top;
    uint64_t** m_counts;
    uint64_t** m_timestamps;
};


} // namespace xmrig


#endif /* XMRIG_HASHRATE_H */
