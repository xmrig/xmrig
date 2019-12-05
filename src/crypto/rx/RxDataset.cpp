/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
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


#include "crypto/rx/RxDataset.h"
#include "backend/common/Tags.h"
#include "base/io/log/Log.h"
#include "crypto/common/VirtualMemory.h"
#include "crypto/rx/RxAlgo.h"
#include "crypto/rx/RxCache.h"


#include <thread>
#include <uv.h>


static_assert(RANDOMX_FLAG_LARGE_PAGES == 1, "RANDOMX_FLAG_LARGE_PAGES flag mismatch");


xmrig::RxDataset::RxDataset(bool hugePages, bool oneGbPages, bool cache, RxConfig::Mode mode) :
    m_mode(mode)
{
    allocate(hugePages, oneGbPages);

    if (cache) {
        m_cache = new RxCache(hugePages);
    }
}


xmrig::RxDataset::RxDataset(RxCache *cache) :
    m_cache(cache)
{
}


xmrig::RxDataset::~RxDataset()
{
    if (m_dataset) {
        randomx_release_dataset(m_dataset);
    }

    delete m_cache;
}


bool xmrig::RxDataset::init(const Buffer &seed, uint32_t numThreads)
{
    if (!m_cache) {
        return false;
    }

    m_cache->init(seed);

    if (!get()) {
        return true;
    }

    const uint64_t datasetItemCount = randomx_dataset_item_count();

    if (numThreads > 1) {
        std::vector<std::thread> threads;
        threads.reserve(numThreads);

        for (uint64_t i = 0; i < numThreads; ++i) {
            const uint32_t a = (datasetItemCount * i) / numThreads;
            const uint32_t b = (datasetItemCount * (i + 1)) / numThreads;
            threads.emplace_back(randomx_init_dataset, m_dataset, m_cache->get(), a, b - a);
        }

        for (uint32_t i = 0; i < numThreads; ++i) {
            threads[i].join();
        }
    }
    else {
        randomx_init_dataset(m_dataset, m_cache->get(), 0, datasetItemCount);
    }

    return true;
}


size_t xmrig::RxDataset::size(bool cache) const
{
    size_t size = 0;

    if (m_dataset) {
        size += maxSize();
    }

    if (cache && m_cache) {
        size += RxCache::maxSize();
    }

    return size;
}


std::pair<uint32_t, uint32_t> xmrig::RxDataset::hugePages(bool cache) const
{
    constexpr size_t twoMiB     = 2U * 1024U * 1024U;
    constexpr size_t oneGiB     = 1024U * 1024U * 1024U;
    constexpr size_t cacheSize  = VirtualMemory::align(RxCache::maxSize(), twoMiB) / twoMiB;
    size_t datasetPageSize      = isOneGbPages() ? oneGiB : twoMiB;
    size_t total                = VirtualMemory::align(maxSize(), datasetPageSize) / datasetPageSize;

    uint32_t count = 0;
    if (isHugePages() || isOneGbPages()) {
        count += total;
    }

    if (cache && m_cache) {
        total += cacheSize;

        if (m_cache->isHugePages()) {
            count += cacheSize;
        }
    }

    return { count, total };
}


void *xmrig::RxDataset::raw() const
{
    return m_dataset ? randomx_get_dataset_memory(m_dataset) : nullptr;
}


void xmrig::RxDataset::setRaw(const void *raw)
{
    if (!m_dataset) {
        return;
    }

    memcpy(randomx_get_dataset_memory(m_dataset), raw, maxSize());
}


void xmrig::RxDataset::allocate(bool hugePages, bool oneGbPages)
{
    if (m_mode == RxConfig::LightMode) {
        LOG_ERR(CLEAR "%s" RED_BOLD_S "fast RandomX mode disabled by config", rx_tag());

        return;
    }

    if (m_mode == RxConfig::AutoMode && uv_get_total_memory() < (maxSize() + RxCache::maxSize())) {
        LOG_ERR(CLEAR "%s" RED_BOLD_S "not enough memory for RandomX dataset", rx_tag());

        return;
    }

    if (hugePages) {
        m_flags   = oneGbPages ? RANDOMX_FLAG_1GB_PAGES : RANDOMX_FLAG_LARGE_PAGES;
        m_dataset = randomx_alloc_dataset(static_cast<randomx_flags>(m_flags));

        if (oneGbPages && !m_dataset) {
            LOG_ERR(CLEAR "%s" RED_BOLD_S "Failed to allocate RandomX dataset using 1GB pages", rx_tag());
            m_flags = RANDOMX_FLAG_LARGE_PAGES;
            m_dataset = randomx_alloc_dataset(static_cast<randomx_flags>(m_flags));
        }
    }

    if (!m_dataset) {
        m_flags   = RANDOMX_FLAG_DEFAULT;
        m_dataset = randomx_alloc_dataset(static_cast<randomx_flags>(m_flags));
    }
}
