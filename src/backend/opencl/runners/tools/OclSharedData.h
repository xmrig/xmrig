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

#ifndef XMRIG_OCLSHAREDDATA_H
#define XMRIG_OCLSHAREDDATA_H


#include <memory>
#include <mutex>


using cl_context = struct _cl_context *;
using cl_mem     = struct _cl_mem *;


namespace xmrig {


class Job;


class OclSharedData
{
public:
    OclSharedData() = default;

    cl_mem createBuffer(cl_context context, size_t size, size_t &offset, size_t limit);
    uint64_t adjustDelay(size_t id);
    uint64_t resumeDelay(size_t id);
    void release();
    void setResumeCounter(uint32_t value);
    void setRunTime(uint64_t time);

    inline size_t threads() const       { return m_threads; }

    inline OclSharedData &operator++()  { ++m_threads; return *this; }

#   ifdef XMRIG_ALGO_RANDOMX
    cl_mem dataset() const;
    void createDataset(cl_context ctx, const Job &job, bool host);
#   endif

private:
    cl_mem m_buffer           = nullptr;
    double m_averageRunTime   = 0.0;
    double m_threshold        = 0.95;
    size_t m_offset           = 0;
    size_t m_threads          = 0;
    std::mutex m_mutex;
    uint32_t m_resumeCounter  = 0;
    uint64_t m_timestamp      = 0;

#   ifdef XMRIG_ALGO_RANDOMX
    cl_mem m_dataset          = nullptr;
#   endif
};


} /* namespace xmrig */


#endif /* XMRIG_OCLSHAREDDATA_H */
