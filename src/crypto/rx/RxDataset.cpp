/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
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


#include "crypto/rx/RxDataset.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/kernel/Platform.h"
#include "crypto/common/VirtualMemory.h"
#include "crypto/rx/RxAlgo.h"
#include "crypto/rx/RxCache.h"


#include <thread>
#include <uv.h>


namespace xmrig {


static void init_dataset_wrapper(randomx_dataset *dataset, randomx_cache *cache, unsigned long startItem, unsigned long itemCount, int priority)
{
    Platform::setThreadPriority(priority);

    randomx_init_dataset(dataset, cache, startItem, itemCount);
}


} // namespace xmrig


xmrig::RxDataset::RxDataset(bool hugePages, bool oneGbPages, bool cache, RxConfig::Mode mode, uint32_t node) :
    m_mode(mode),
    m_node(node)
{
    allocate(hugePages, oneGbPages);

    if (isOneGbPages()) {
        m_cache = new RxCache(m_memory->raw() + VirtualMemory::align(maxSize()));

        return;
    }

    if (cache) {
        m_cache = new RxCache(hugePages, node);
    }
}


xmrig::RxDataset::RxDataset(RxCache *cache) :
    m_node(0),
    m_cache(cache)
{
}


xmrig::RxDataset::~RxDataset()
{
    randomx_release_dataset(m_dataset);

    delete m_cache;
    delete m_memory;
}


bool xmrig::RxDataset::init(const Buffer &seed, uint32_t numThreads, int priority)
{
    if (!m_cache || !m_cache->get()) {
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
            threads.emplace_back(init_dataset_wrapper, m_dataset, m_cache->get(), a, b - a, priority);
        }

        for (uint32_t i = 0; i < numThreads; ++i) {
            threads[i].join();
        }
    }
    else {
        init_dataset_wrapper(m_dataset, m_cache->get(), 0, datasetItemCount, priority);
    }

    return true;
}


bool xmrig::RxDataset::isHugePages() const
{
    return m_memory && m_memory->isHugePages();
}


bool xmrig::RxDataset::isOneGbPages() const
{
    return m_memory && m_memory->isOneGbPages();
}


xmrig::HugePagesInfo xmrig::RxDataset::hugePages(bool cache) const
{
    auto pages = m_memory ? m_memory->hugePages() : HugePagesInfo();

    if (cache && m_cache) {
        pages += m_cache->hugePages();
    }

    return pages;
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
        LOG_ERR(CLEAR "%s" RED_BOLD_S "fast RandomX mode disabled by config", Tags::randomx());

        return;
    }

    if (m_mode == RxConfig::AutoMode && uv_get_total_memory() < (maxSize() + RxCache::maxSize())) {
        LOG_ERR(CLEAR "%s" RED_BOLD_S "not enough memory for RandomX dataset", Tags::randomx());

        return;
    }

    m_memory  = new VirtualMemory(maxSize(), hugePages, oneGbPages, false, m_node);

    if (m_memory->isOneGbPages()) {
        m_scratchpadOffset = maxSize() + RANDOMX_CACHE_MAX_SIZE;
        m_scratchpadLimit = m_memory->capacity();
    }

    m_dataset = randomx_create_dataset(m_memory->raw());

#   ifdef XMRIG_OS_LINUX
    if (oneGbPages && !isOneGbPages()) {
        LOG_ERR(CLEAR "%s" RED_BOLD_S "failed to allocate RandomX dataset using 1GB pages", Tags::randomx());
    }
#   endif
}


uint8_t* xmrig::RxDataset::tryAllocateScrathpad()
{
    uint8_t* p = reinterpret_cast<uint8_t*>(raw());
    if (!p) {
        return nullptr;
    }

    const size_t offset = m_scratchpadOffset.fetch_add(RANDOMX_SCRATCHPAD_L3_MAX_SIZE);
    if (offset + RANDOMX_SCRATCHPAD_L3_MAX_SIZE > m_scratchpadLimit) {
        return nullptr;
    }

    return p + offset;
}
