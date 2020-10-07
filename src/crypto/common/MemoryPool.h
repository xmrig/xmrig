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

#ifndef XMRIG_MEMORYPOOL_H
#define XMRIG_MEMORYPOOL_H


#include "backend/common/interfaces/IMemoryPool.h"
#include "base/tools/Object.h"


namespace xmrig {


class VirtualMemory;


class MemoryPool : public IMemoryPool
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(MemoryPool)

    MemoryPool(size_t size, bool hugePages, uint32_t node = 0);
    ~MemoryPool() override;

protected:
    bool isHugePages(uint32_t node) const override;
    uint8_t *get(size_t size, uint32_t node) override;
    void release(uint32_t node) override;

private:
    size_t m_refs           = 0;
    size_t m_offset         = 0;
    size_t m_alignOffset    = 0;
    VirtualMemory *m_memory = nullptr;
};


} /* namespace xmrig */



#endif /* XMRIG_MEMORYPOOL_H */
