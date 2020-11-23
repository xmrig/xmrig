/* XMRig
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

#ifndef XMRIG_BENCHSTATE_H
#define XMRIG_BENCHSTATE_H


#include <cstddef>
#include <cstdint>


namespace xmrig {


class Algorithm;
class IBackend;
class IBenchListener;


class BenchState
{
public:
    static bool isDone();
    static uint64_t referenceHash(const Algorithm &algo, uint32_t size, uint32_t threads);
    static uint64_t start(size_t threads, const IBackend *backend);
    static void destroy();
    static void done(uint64_t data, uint64_t diff, uint64_t ts);

    inline static uint32_t size()                               { return m_size; }
    inline static void setListener(IBenchListener *listener)    { m_listener = listener; }
    inline static void setSize(uint32_t size)                   { m_size = size; }

private:
    static IBenchListener *m_listener;
    static uint32_t m_size;
};


} // namespace xmrig


#endif /* XMRIG_BENCHSTATE_H */
