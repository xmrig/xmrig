/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
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


#include "crypto/common/NUMAMemoryPool.h"
#include "crypto/common/VirtualMemory.h"
#include "backend/cpu/Cpu.h"
#include "crypto/common/MemoryPool.h"


#include <algorithm>


xmrig::NUMAMemoryPool::NUMAMemoryPool(size_t size, bool hugePages) :
    m_hugePages(hugePages),
    m_nodeSize(std::max<size_t>(size / Cpu::info()->nodes(), 1)),
    m_size(size)
{
}


xmrig::NUMAMemoryPool::~NUMAMemoryPool()
{
    for (auto kv : m_map) {
        delete kv.second;
    }
}


bool xmrig::NUMAMemoryPool::isHugePages(uint32_t node) const
{
    if (!m_size) {
        return false;
    }

    return getOrCreate(node)->isHugePages(node);
}


uint8_t *xmrig::NUMAMemoryPool::get(size_t size, uint32_t node)
{
    if (!m_size) {
        return nullptr;
    }

    return getOrCreate(node)->get(size, node);
}


void xmrig::NUMAMemoryPool::release(uint32_t node)
{
    const auto pool = get(node);
    if (pool) {
        pool->release(node);
    }
}


xmrig::IMemoryPool *xmrig::NUMAMemoryPool::get(uint32_t node) const
{
    return m_map.count(node) ? m_map.at(node) : nullptr;
}


xmrig::IMemoryPool *xmrig::NUMAMemoryPool::getOrCreate(uint32_t node) const
{
    auto pool = get(node);
    if (!pool) {
        pool = new MemoryPool(m_nodeSize, m_hugePages, node);
        m_map.insert({ node, pool });
    }

    return pool;
}
