/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include "backend/opencl/runners/tools/OclSharedData.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/tools/Chrono.h"
#include "crypto/rx/Rx.h"
#include "crypto/rx/RxDataset.h"


#include <algorithm>
#include <cinttypes>
#include <stdexcept>
#include <thread>


constexpr size_t oneGiB = 1024 * 1024 * 1024;


cl_mem xmrig::OclSharedData::createBuffer(cl_context context, size_t size, size_t &offset, size_t limit)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    const size_t pos = offset + (size * m_offset);
    size             = std::max(size * m_threads, oneGiB);

    if (size > limit) {
        return nullptr;
    }

    offset = pos;
    ++m_offset;

    if (!m_buffer) {
        m_buffer = OclLib::createBuffer(context, CL_MEM_READ_WRITE, size);
    }

    return OclLib::retain(m_buffer);
}


uint64_t xmrig::OclSharedData::adjustDelay(size_t id)
{
    if (m_threads < 2) {
        return 0;
    }

    const uint64_t t0 = Chrono::steadyMSecs();
    uint64_t delay    = 0;

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        const uint64_t dt = t0 - m_timestamp;
        m_timestamp = t0;

        // The perfect interleaving is when N threads on the same GPU start with T/N interval between each other
        // If a thread starts earlier than 0.75*T/N ms after the previous thread, delay it to restore perfect interleaving
        if ((dt > 0) && (dt < m_threshold * (m_averageRunTime / m_threads))) {
            delay = static_cast<uint64_t>(m_averageRunTime / m_threads - dt);
            m_threshold = 0.75;
        }
    }

    if (delay == 0) {
        return 0;
    }

    if (delay >= 400) {
        delay = 200;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(delay));

#   ifdef XMRIG_INTERLEAVE_DEBUG
    LOG_WARN("Thread #%zu was paused for %" PRIu64 " ms to adjust interleaving", id, delay);
#   endif

    return delay;
}


uint64_t xmrig::OclSharedData::resumeDelay(size_t id)
{
    if (m_threads < 2) {
        return 0;
    }

    uint64_t delay = 0;

    {
        constexpr const double firstThreadSpeedupCoeff = 1.25;

        std::lock_guard<std::mutex> lock(m_mutex);
        delay = static_cast<uint64_t>(m_resumeCounter * m_averageRunTime / m_threads / firstThreadSpeedupCoeff);
        ++m_resumeCounter;
    }

    if (delay == 0) {
        return 0;
    }

    if (delay > 1000) {
        delay = 1000;
    }

#   ifdef XMRIG_INTERLEAVE_DEBUG
    LOG_WARN("Thread #%zu will be paused for %" PRIu64 " ms to before resuming", id, delay);
#   endif

    std::this_thread::sleep_for(std::chrono::milliseconds(delay));

    return delay;
}


void xmrig::OclSharedData::release()
{
    OclLib::release(m_buffer);

#   ifdef XMRIG_ALGO_RANDOMX
    OclLib::release(m_dataset);
#   endif
}


void xmrig::OclSharedData::setResumeCounter(uint32_t value)
{
    if (m_threads < 2) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_resumeCounter = value;
}


void xmrig::OclSharedData::setRunTime(uint64_t time)
{
    // averagingBias = 1.0 - only the last delta time is taken into account
    // averagingBias = 0.5 - the last delta time has the same weight as all the previous ones combined
    // averagingBias = 0.1 - the last delta time has 10% weight of all the previous ones combined
    constexpr double averagingBias = 0.1;

    std::lock_guard<std::mutex> lock(m_mutex);
    m_averageRunTime = m_averageRunTime * (1.0 - averagingBias) + time * averagingBias;
}


#ifdef XMRIG_ALGO_RANDOMX
cl_mem xmrig::OclSharedData::dataset() const
{
    if (!m_dataset) {
        throw std::runtime_error("RandomX dataset is not available");
    }

    return OclLib::retain(m_dataset);
}


void xmrig::OclSharedData::createDataset(cl_context ctx, const Job &job, bool host)
{
    if (m_dataset) {
        return;
    }

    cl_int ret;

    if (host) {
        auto dataset = Rx::dataset(job, 0);

        m_dataset = OclLib::createBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, RxDataset::maxSize(), dataset->raw(), &ret);
    }
    else {
        m_dataset = OclLib::createBuffer(ctx, CL_MEM_READ_ONLY, RxDataset::maxSize(), nullptr, &ret);
    }
}
#endif
