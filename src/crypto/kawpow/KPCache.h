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

#ifndef XMRIG_KP_CACHE_H
#define XMRIG_KP_CACHE_H


#include "base/tools/Object.h"
#include <mutex>
#include <vector>


namespace xmrig
{


class VirtualMemory;


class KPCache
{
public:
    static constexpr size_t l1_cache_size = 16 * 1024;
    static constexpr size_t l1_cache_num_items = l1_cache_size / sizeof(uint32_t);
    static constexpr uint32_t num_dataset_parents = 512;

    XMRIG_DISABLE_COPY_MOVE(KPCache)

    KPCache();
    ~KPCache();

    bool init(uint32_t epoch);

    void* data() const;
    size_t size() const { return m_size; }
    uint32_t epoch() const { return m_epoch; }

    const uint32_t* l1_cache() const { return m_DAGCache.data(); }

    static uint64_t cache_size(uint32_t epoch);
    static uint64_t dag_size(uint32_t epoch);

    static void calculate_fast_mod_data(uint32_t divisor, uint32_t &reciprocal, uint32_t &increment, uint32_t& shift);

    static std::mutex s_cacheMutex;
    static KPCache s_cache;

private:
    VirtualMemory* m_memory = nullptr;
    size_t m_size = 0;
    uint32_t m_epoch = 0xFFFFFFFFUL;
    std::vector<uint32_t> m_DAGCache;
};


} /* namespace xmrig */


#endif /* XMRIG_KP_CACHE_H */
