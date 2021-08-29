/* XMRig
 * Copyright (c) 2018-2019 tevador     <tevador@gmail.com>
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


#include "crypto/rx/RxBasicStorage.h"
#include "backend/common/Tags.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/tools/Chrono.h"
#include "crypto/rx/RxAlgo.h"
#include "crypto/rx/RxCache.h"
#include "crypto/rx/RxDataset.h"
#include "crypto/rx/RxSeed.h"


namespace xmrig {


constexpr size_t oneMiB = 1024 * 1024;


class RxBasicStoragePrivate
{
public:
    XMRIG_DISABLE_COPY_MOVE(RxBasicStoragePrivate)

    inline RxBasicStoragePrivate() = default;
    inline ~RxBasicStoragePrivate() { deleteDataset(); }

    inline bool isReady(const Job &job) const   { return m_ready && m_seed == job; }
    inline RxDataset *dataset() const           { return m_dataset; }
    inline void deleteDataset()                 { delete m_dataset; m_dataset = nullptr; }


    inline void setSeed(const RxSeed &seed)
    {
        m_ready = false;

        if (m_seed.algorithm() != seed.algorithm()) {
            RxAlgo::apply(seed.algorithm());
        }

        m_seed = seed;
    }


    inline bool createDataset(bool hugePages, bool oneGbPages, RxConfig::Mode mode)
    {
        const uint64_t ts = Chrono::steadyMSecs();

        m_dataset = new RxDataset(hugePages, oneGbPages, true, mode, 0);
        if (!m_dataset->cache()->get()) {
            deleteDataset();

            LOG_INFO("%s" RED_BOLD("failed to allocate RandomX memory") BLACK_BOLD(" (%" PRIu64 " ms)"), Tags::randomx(), Chrono::steadyMSecs() - ts);

            return false;
        }

        printAllocStatus(ts);

        return true;
    }


    inline void initDataset(uint32_t threads, int priority)
    {
        const uint64_t ts = Chrono::steadyMSecs();

        m_ready = m_dataset->init(m_seed.data(), threads, priority);

        if (m_ready) {
            LOG_INFO("%s" GREEN_BOLD("dataset ready") BLACK_BOLD(" (%" PRIu64 " ms)"), Tags::randomx(), Chrono::steadyMSecs() - ts);
        }
    }


private:
    void printAllocStatus(uint64_t ts)
    {
        if (m_dataset->get() != nullptr) {
            const auto pages = m_dataset->hugePages();

            LOG_INFO("%s" GREEN_BOLD("allocated") CYAN_BOLD(" %zu MB") BLACK_BOLD(" (%zu+%zu)") " huge pages %s%1.0f%% %u/%u" CLEAR " %sJIT" BLACK_BOLD(" (%" PRIu64 " ms)"),
                     Tags::randomx(),
                     pages.size / oneMiB,
                     RxDataset::maxSize() / oneMiB,
                     RxCache::maxSize() / oneMiB,
                     (pages.isFullyAllocated() ? GREEN_BOLD_S : (pages.allocated == 0 ? RED_BOLD_S : YELLOW_BOLD_S)),
                     pages.percent(),
                     pages.allocated,
                     pages.total,
                     m_dataset->cache()->isJIT() ? GREEN_BOLD_S "+" : RED_BOLD_S "-",
                     Chrono::steadyMSecs() - ts
                     );
        }
        else {
            LOG_WARN(CLEAR "%s" YELLOW_BOLD_S "failed to allocate RandomX dataset, switching to slow mode" BLACK_BOLD(" (%" PRIu64 " ms)"), Tags::randomx(), Chrono::steadyMSecs() - ts);
        }
    }


    bool m_ready         = false;
    RxDataset *m_dataset = nullptr;
    RxSeed m_seed;
};


} // namespace xmrig


xmrig::RxBasicStorage::RxBasicStorage() :
    d_ptr(new RxBasicStoragePrivate())
{
}


xmrig::RxBasicStorage::~RxBasicStorage()
{
    delete d_ptr;
}


bool xmrig::RxBasicStorage::isAllocated() const
{
    return d_ptr->dataset() && d_ptr->dataset()->cache() && d_ptr->dataset()->cache()->get();
}


xmrig::HugePagesInfo xmrig::RxBasicStorage::hugePages() const
{
    if (!d_ptr->dataset()) {
        return {};
    }

    return d_ptr->dataset()->hugePages();
}


xmrig::RxDataset *xmrig::RxBasicStorage::dataset(const Job &job, uint32_t) const
{
    if (!d_ptr->isReady(job)) {
        return nullptr;
    }

    return d_ptr->dataset();
}


void xmrig::RxBasicStorage::init(const RxSeed &seed, uint32_t threads, bool hugePages, bool oneGbPages, RxConfig::Mode mode, int priority)
{
    d_ptr->setSeed(seed);

    if (!d_ptr->dataset() && !d_ptr->createDataset(hugePages, oneGbPages, mode)) {
        return;
    }

    d_ptr->initDataset(threads, priority);
}
