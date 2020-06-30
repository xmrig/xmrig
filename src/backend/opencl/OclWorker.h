/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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

#ifndef XMRIG_OCLWORKER_H
#define XMRIG_OCLWORKER_H


#include "backend/common/Worker.h"
#include "backend/common/WorkerJob.h"
#include "backend/opencl/OclLaunchData.h"
#include "base/tools/Object.h"
#include "net/JobResult.h"


namespace xmrig {


class IOclRunner;
class Job;


class OclWorker : public Worker
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(OclWorker)

    OclWorker(size_t id, const OclLaunchData &data);

    ~OclWorker() override;

    void jobEarlyNotification(const Job&) override;

    static std::atomic<bool> ready;

protected:
    bool selfTest() override;
    size_t intensity() const override;
    void start() override;

private:
    bool consumeJob();
    void storeStats(uint64_t ts);

    const Algorithm m_algorithm;
    const Miner *m_miner;
    const uint32_t m_intensity;
    IOclRunner *m_runner = nullptr;
    OclSharedData &m_sharedData;
    WorkerJob<1> m_job;
    uint32_t m_deviceIndex;
};


} // namespace xmrig


#endif /* XMRIG_OCLWORKER_H */
