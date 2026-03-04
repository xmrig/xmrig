/* XMRig
 * Copyright (c) 2018-2026 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2026 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_WRITEBATON_H
#define XMRIG_WRITEBATON_H


#include <cstddef>
#include <vector>
#include <uv.h>


namespace xmrig {


class WriteBaton
{
public:
    explicit WriteBaton(const char *data, size_t size) :
        storage(data, data + size),
        buf(uv_buf_init(storage.data(), static_cast<unsigned int>(storage.size())))
    {
        req.data = this;
    }

    uv_write_t req{};
    std::vector<char> storage;
    uv_buf_t buf;
};


} /* namespace xmrig */


#endif /* XMRIG_WRITEBATON_H */
