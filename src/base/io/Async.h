/* XMRig
 * Copyright (c) 2015-2020 libuv project contributors.
 * Copyright (c) 2020      cohcho      <https://github.com/cohcho>
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

#ifndef XMRIG_ASYNC_H
#define XMRIG_ASYNC_H


#include "base/tools/Object.h"


namespace xmrig {


class AsyncPrivate;
class IAsyncListener;


class Async
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(Async)

    Async(IAsyncListener *listener);
    ~Async();

    void send();

private:
    AsyncPrivate *d_ptr;
};


} // namespace xmrig


#endif /* XMRIG_ASYNC_H */
