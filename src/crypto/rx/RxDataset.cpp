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


#include <thread>


#include "crypto/common/VirtualMemory.h"
#include "crypto/randomx/randomx.h"
#include "crypto/rx/RxAlgo.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"


static_assert(RANDOMX_FLAG_LARGE_PAGES == 1, "RANDOMX_FLAG_LARGE_PAGES flag mismatch");


xmrig::RxDataset::RxDataset(bool hugePages)
{
    if (hugePages) {
        m_flags   = RANDOMX_FLAG_LARGE_PAGES;
        m_dataset = randomx_alloc_dataset(static_cast<randomx_flags>(m_flags));
    }

    if (!m_dataset) {
        m_flags   = RANDOMX_FLAG_DEFAULT;
        m_dataset = randomx_alloc_dataset(static_cast<randomx_flags>(m_flags));
    }

    m_cache = new RxCache(hugePages);
}


xmrig::RxDataset::~RxDataset()
{
    if (m_dataset) {
        randomx_release_dataset(m_dataset);
    }

    delete m_cache;
}


bool xmrig::RxDataset::init(const void *seed, const Algorithm &algorithm, uint32_t numThreads)
{
    if (isReady(seed, algorithm)) {
        return false;
    }

    if (m_algorithm != algorithm) {
        m_algorithm = RxAlgo::apply(algorithm);
    }

    cache()->init(seed);

    if (!get()) {
        return true;
    }

    const uint32_t datasetItemCount = randomx_dataset_item_count();

    if (numThreads > 1) {
        std::vector<std::thread> threads;
        threads.reserve(numThreads);

        for (uint32_t i = 0; i < numThreads; ++i) {
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


bool xmrig::RxDataset::isReady(const void *seed, const Algorithm &algorithm) const
{
    return algorithm == m_algorithm && cache()->isReady(seed);
}


std::pair<size_t, size_t> xmrig::RxDataset::hugePages() const
{
    constexpr size_t twoMiB = 2u * 1024u * 1024u;
    constexpr const size_t total = (VirtualMemory::align(size(), twoMiB) + VirtualMemory::align(RxCache::size(), twoMiB)) / twoMiB;

    size_t count = 0;
    if (isHugePages()) {
        count += VirtualMemory::align(size(), twoMiB) / twoMiB;
    }

    if (m_cache->isHugePages()) {
        count += VirtualMemory::align(RxCache::size(), twoMiB) / twoMiB;
    }

    return std::pair<size_t, size_t>(count, total);
}
