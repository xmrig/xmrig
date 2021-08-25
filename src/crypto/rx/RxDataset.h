/* XMRig
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_RX_DATASET_H
#define XMRIG_RX_DATASET_H


#include "base/tools/Buffer.h"
#include "base/tools/Object.h"
#include "crypto/common/HugePagesInfo.h"
#include "crypto/randomx/configuration.h"
#include "crypto/rx/RxConfig.h"

#include <atomic>


struct randomx_dataset;


namespace xmrig
{


class RxCache;
class VirtualMemory;


class RxDataset
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(RxDataset)

    RxDataset(bool hugePages, bool oneGbPages, bool cache, RxConfig::Mode mode, uint32_t node);
    RxDataset(RxCache *cache);
    ~RxDataset();

    inline randomx_dataset *get() const     { return m_dataset; }
    inline RxCache *cache() const           { return m_cache; }
    inline void setCache(RxCache *cache)    { m_cache = cache; }

    bool init(const Buffer &seed, uint32_t numThreads, int priority);
    bool isHugePages() const;
    bool isOneGbPages() const;
    HugePagesInfo hugePages(bool cache = true) const;
    size_t size(bool cache = true) const;
    uint8_t *tryAllocateScrathpad();
    void *raw() const;
    void setRaw(const void *raw);

    static inline constexpr size_t maxSize() { return RANDOMX_DATASET_MAX_SIZE; }

private:
    void allocate(bool hugePages, bool oneGbPages);

    const RxConfig::Mode m_mode = RxConfig::FastMode;
    const uint32_t m_node;
    randomx_dataset *m_dataset  = nullptr;
    RxCache *m_cache            = nullptr;
    size_t m_scratchpadLimit    = 0;
    std::atomic<size_t> m_scratchpadOffset{};
    VirtualMemory *m_memory     = nullptr;
};


} /* namespace xmrig */


#endif /* XMRIG_RX_DATASET_H */
