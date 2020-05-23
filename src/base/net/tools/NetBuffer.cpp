/* XMRig
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


#include "base/net/tools/MemPool.h"
#include "base/net/tools/NetBuffer.h"


#include <cassert>
#include <uv.h>


namespace xmrig {


static constexpr size_t kInitSize                       = 4;
static MemPool<NetBuffer::kChunkSize, kInitSize> *pool  = nullptr;


inline MemPool<NetBuffer::kChunkSize, kInitSize> *getPool()
{
    if (!pool) {
        pool = new MemPool<NetBuffer::kChunkSize, kInitSize>();
    }

    return pool;
}


} // namespace xmrig


char *xmrig::NetBuffer::allocate()
{
    return getPool()->allocate();
}


void xmrig::NetBuffer::destroy()
{
    if (!pool) {
        return;
    }

    assert(pool->freeSize() == pool->size());

    delete pool;
    pool = nullptr;
}


void xmrig::NetBuffer::onAlloc(uv_handle_t *, size_t, uv_buf_t *buf)
{
    buf->base = getPool()->allocate();
    buf->len  = kChunkSize;
}


void xmrig::NetBuffer::release(const char *buf)
{
    getPool()->deallocate(buf);
}


void xmrig::NetBuffer::release(const uv_buf_t *buf)
{
    getPool()->deallocate(buf->base);
}
