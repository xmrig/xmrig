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


#include "backend/cuda/runners/CudaBaseRunner.h"
#include "backend/cuda/wrappers/CudaLib.h"
#include "backend/cuda/CudaLaunchData.h"
#include "backend/common/Tags.h"
#include "base/io/log/Log.h"
#include "base/net/stratum/Job.h"


xmrig::CudaBaseRunner::CudaBaseRunner(size_t id, const CudaLaunchData &data) :
    m_data(data),
    m_threadId(id)
{
}


xmrig::CudaBaseRunner::~CudaBaseRunner()
{
    CudaLib::release(m_ctx);
}


bool xmrig::CudaBaseRunner::init()
{
    m_ctx = CudaLib::alloc(m_data.thread.index(), m_data.thread.bfactor(), m_data.thread.bsleep());
    if (!callWrapper(CudaLib::deviceInfo(m_ctx, m_data.thread.blocks(), m_data.thread.threads(), m_data.algorithm, m_data.thread.datasetHost()))) {
        return false;
    }

    return callWrapper(CudaLib::deviceInit(m_ctx));
}


bool xmrig::CudaBaseRunner::set(const Job &job, uint8_t *blob)
{
    m_height = job.height();
    m_target = job.target();

    return callWrapper(CudaLib::setJob(m_ctx, blob, job.size(), job.algorithm()));
}


size_t xmrig::CudaBaseRunner::intensity() const
{
    return m_data.thread.threads() * m_data.thread.blocks();
}


bool xmrig::CudaBaseRunner::callWrapper(bool result) const
{
    if (!result) {
        const char *error = CudaLib::lastError(m_ctx);
        if (error) {
            LOG_ERR("%s" RED_S " thread " RED_BOLD("#%zu") RED_S " failed with error " RED_BOLD("%s"), cuda_tag(), m_threadId, error);
        }
    }

    return result;
}
