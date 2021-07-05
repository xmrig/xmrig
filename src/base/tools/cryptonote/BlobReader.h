/* XMRig
 * Copyright 2012-2013 The Cryptonote developers
 * Copyright 2014-2021 The Monero Project
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_BLOBREADER_H
#define XMRIG_BLOBREADER_H


#include <cstdint>


namespace xmrig {


class CBlobReader
{
public:
    inline CBlobReader(const void* data, size_t size)
        : m_data(reinterpret_cast<const uint8_t*>(data))
        , m_size(size)
        , m_index(0)
    {}

    inline bool operator()(uint8_t& data) { return getByte(data); }
    inline bool operator()(uint64_t& data) { return getVarint(data); }

    template<size_t N>
    inline bool operator()(uint8_t(&data)[N])
    {
        for (size_t i = 0; i < N; ++i) {
            if (!getByte(data[i])) {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    inline void readItems(T& data, size_t count)
    {
        data.resize(count);
        for (size_t i = 0; i < count; ++i)
            operator()(data[i]);
    }

    inline size_t index() const { return m_index; }

    inline void skip(size_t N) { m_index += N; }

private:
    inline bool getByte(uint8_t& data)
    {
        if (m_index >= m_size) {
            return false;
        }

        data = m_data[m_index++];
        return true;
    }

    inline bool getVarint(uint64_t& data)
    {
        uint64_t result = 0;
        uint8_t t;
        int shift = 0;

        do {
            if (!getByte(t)) {
                return false;
            }
            result |= static_cast<uint64_t>(t & 0x7F) << shift;
            shift += 7;
        } while (t & 0x80);

        data = result;
        return true;
    }

    const uint8_t* m_data;
    size_t m_size;
    size_t m_index;
};


} /* namespace xmrig */


#endif /* XMRIG_BLOBREADER_H */
