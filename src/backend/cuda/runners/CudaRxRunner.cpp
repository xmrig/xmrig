/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2024 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "backend/cuda/runners/CudaRxRunner.h"
#include "backend/cuda/CudaLaunchData.h"
#include "backend/cuda/wrappers/CudaLib.h"
#include "base/net/stratum/Job.h"
#include "crypto/rx/Rx.h"
#include "crypto/rx/RxDataset.h"


xmrig::CudaRxRunner::CudaRxRunner(size_t index, const CudaLaunchData &data) :
    CudaBaseRunner(index, data),
    m_datasetHost(data.thread.datasetHost() > 0)
{
    m_intensity                   = m_data.thread.threads() * m_data.thread.blocks();
    const size_t scratchpads_size = m_intensity * m_data.algorithm.l3();
    const size_t num_scratchpads  = scratchpads_size / m_data.algorithm.l3();

    if (m_intensity > num_scratchpads) {
        m_intensity = num_scratchpads;
    }

    m_intensity -= m_intensity % 32;
}


bool xmrig::CudaRxRunner::run(uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce)
{
    return callWrapper(CudaLib::rxHash(m_ctx, startNonce, m_target, rescount, resnonce));
}


bool xmrig::CudaRxRunner::set(const Job &job, uint8_t *blob)
{
    if (!m_datasetHost && (m_seed != job.seed())) {
        m_seed = job.seed();

        if (m_ready) {
            const auto *dataset = Rx::dataset(job, 0);
            callWrapper(CudaLib::rxUpdateDataset(m_ctx, dataset->raw(), dataset->size(false)));
        }
    }

    const bool rc = CudaBaseRunner::set(job, blob);
    if (!rc || m_ready) {
        return rc;
    }

    const auto *dataset = Rx::dataset(job, 0);
    m_ready = callWrapper(CudaLib::rxPrepare(m_ctx, dataset->raw(), dataset->size(false), m_datasetHost, m_intensity));

    return m_ready;
}
