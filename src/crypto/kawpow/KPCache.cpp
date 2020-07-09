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


#include <cinttypes>
#include <algorithm>
#include <thread>

#include "crypto/kawpow/KPCache.h"
#include "3rdparty/libethash/data_sizes.h"
#include "3rdparty/libethash/ethash_internal.h"
#include "3rdparty/libethash/ethash.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/tools/Chrono.h"
#include "crypto/common/VirtualMemory.h"


namespace xmrig {


std::mutex KPCache::s_cacheMutex;
KPCache KPCache::s_cache;


KPCache::KPCache()
{
}


KPCache::~KPCache()
{
    delete m_memory;
}


bool KPCache::init(uint32_t epoch)
{
    if (epoch >= sizeof(cache_sizes) / sizeof(cache_sizes[0])) {
        return false;
    }

    if (m_epoch == epoch) {
        return true;
    }

    const uint64_t start_ms = Chrono::steadyMSecs();

    const size_t size = cache_sizes[epoch];
    if (!m_memory || m_memory->size() < size) {
        delete m_memory;
        m_memory = new VirtualMemory(size, false, false, false);
    }

    const ethash_h256_t seedhash = ethash_get_seedhash(epoch);
    ethash_compute_cache_nodes(m_memory->raw(), size, &seedhash);

    ethash_light cache;
    cache.cache = m_memory->raw();
    cache.cache_size = size;

    cache.num_parent_nodes = cache.cache_size / sizeof(node);
    calculate_fast_mod_data(cache.num_parent_nodes, cache.reciprocal, cache.increment, cache.shift);

    const uint64_t cache_nodes = (size + sizeof(node) * 4 - 1) / sizeof(node);
    m_DAGCache.resize(cache_nodes * (sizeof(node) / sizeof(uint32_t)));

    // Init DAG cache
    {
        const uint64_t n = std::max(std::thread::hardware_concurrency(), 1U);

        std::vector<std::thread> threads;
        threads.reserve(n);

        for (uint64_t i = 0; i < n; ++i) {
            const uint32_t a = (cache_nodes * i) / n;
            const uint32_t b = (cache_nodes * (i + 1)) / n;

            threads.emplace_back([this, a, b, cache_nodes, &cache]() {
                uint32_t j = a;
                for (; j + 4 <= b; j += 4) ethash_calculate_dag_item4_opt(((node*)m_DAGCache.data()) + j, j, num_dataset_parents, &cache);
                for (; j < b; ++j) ethash_calculate_dag_item_opt(((node*)m_DAGCache.data()) + j, j, num_dataset_parents, &cache);
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    m_size = size;
    m_epoch = epoch;

    LOG_INFO("%s " YELLOW("KawPow") " light cache for epoch " WHITE_BOLD("%u") " calculated " BLACK_BOLD("(%" PRIu64 "ms)"), Tags::miner(), epoch, Chrono::steadyMSecs() - start_ms);

    return true;
}


void* KPCache::data() const
{
    return m_memory ? m_memory->raw() : nullptr;
}


static inline uint32_t clz(uint32_t a)
{
#ifdef _MSC_VER
    unsigned long index;
    _BitScanReverse(&index, a);
    return 31 - index;
#else
    return __builtin_clz(a);
#endif
}


uint64_t KPCache::cache_size(uint32_t epoch)
{
    if (epoch >= sizeof(cache_sizes) / sizeof(cache_sizes[0])) {
        return 0;
    }

    return cache_sizes[epoch];
}


uint64_t KPCache::dag_size(uint32_t epoch)
{
    if (epoch >= sizeof(dag_sizes) / sizeof(dag_sizes[0])) {
        return 0;
    }

    return dag_sizes[epoch];
}


void KPCache::calculate_fast_mod_data(uint32_t divisor, uint32_t& reciprocal, uint32_t& increment, uint32_t& shift)
{
    if ((divisor & (divisor - 1)) == 0) {
        reciprocal = 1;
        increment = 0;
        shift = 31U - clz(divisor);
    }
    else {
        shift = 63U - clz(divisor);
        const uint64_t N = 1ULL << shift;
        const uint64_t q = N / divisor;
        const uint64_t r = N - q * divisor;
        if (r * 2 < divisor)
        {
            reciprocal = static_cast<uint32_t>(q);
            increment = 1;
        }
        else
        {
            reciprocal = static_cast<uint32_t>(q + 1);
            increment = 0;
        }
    }
}


} // namespace xmrig
