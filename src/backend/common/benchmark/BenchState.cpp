/* XMRig
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


#include "backend/common/benchmark/BenchState.h"
#include "backend/common/benchmark/BenchState_test.h"
#include "backend/common/interfaces/IBenchListener.h"
#include "base/io/Async.h"
#include "base/tools/Chrono.h"


#include <algorithm>
#include <cassert>
#include <memory>
#include <mutex>


namespace xmrig {


static bool done                        = false;
static std::mutex mutex;
static std::shared_ptr<Async> async;
static uint32_t remaining               = 0;
static uint64_t doneTime                = 0;
static uint64_t result                  = 0;
static uint64_t topDiff                 = 0;


IBenchListener *BenchState::m_listener  = nullptr;
uint32_t BenchState::m_size             = 0;


} // namespace xmrig



bool xmrig::BenchState::isDone()
{
    return xmrig::done;
}


uint64_t xmrig::BenchState::referenceHash(const Algorithm &algo, uint32_t size, uint32_t threads)
{
    uint64_t hash = 0;

    try {
        const auto &h = (threads == 1) ? hashCheck1T : hashCheck;
        hash = h.at(algo).at(size);
    } catch (const std::exception &ex) {}

    return hash;
}


uint64_t xmrig::BenchState::start(size_t threads, const IBackend *backend)
{
    assert(m_listener != nullptr);

    remaining = static_cast<uint32_t>(threads);

    async = std::make_shared<Async>([] {
        m_listener->onBenchDone(result, topDiff, doneTime);
        async.reset();
        xmrig::done = true;
    });

    const uint64_t ts = Chrono::steadyMSecs();
    m_listener->onBenchReady(ts, remaining, backend);

    return ts;
}


void xmrig::BenchState::destroy()
{
    xmrig::done = true;
    async.reset();
}


void xmrig::BenchState::done(uint64_t data, uint64_t diff, uint64_t ts)
{
    assert(async && remaining > 0);

    std::lock_guard<std::mutex> lock(mutex);

    result ^= data;
    doneTime = std::max(doneTime, ts);
    topDiff  = std::max(topDiff, diff);
    --remaining;

    if (remaining == 0) {
        async->send();
    }
}
