/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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


#include "backend/cuda/runners/CudaAstroBWTRunner.h"
#include "backend/cuda/CudaLaunchData.h"
#include "backend/cuda/wrappers/CudaLib.h"
#include "base/net/stratum/Job.h"


constexpr uint32_t xmrig::CudaAstroBWTRunner::BWT_DATA_STRIDE;


xmrig::CudaAstroBWTRunner::CudaAstroBWTRunner(size_t index, const CudaLaunchData &data) :
    CudaBaseRunner(index, data)
{
    m_intensity = m_data.thread.threads() * m_data.thread.blocks();
    m_intensity -= m_intensity % 32;
}


bool xmrig::CudaAstroBWTRunner::run(uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce)
{
    return callWrapper(CudaLib::astroBWTHash(m_ctx, startNonce, m_target, rescount, resnonce));
}


bool xmrig::CudaAstroBWTRunner::set(const Job &job, uint8_t *blob)
{
    if (!CudaBaseRunner::set(job, blob)) {
        return false;
    }

    return callWrapper(CudaLib::astroBWTPrepare(m_ctx, static_cast<uint32_t>(m_intensity)));
}


size_t xmrig::CudaAstroBWTRunner::roundSize() const
{
    constexpr uint32_t STAGE1_SIZE = 147253;
    constexpr uint32_t STAGE1_DATA_STRIDE = (STAGE1_SIZE + 256 + 255) & ~255U;

    const uint32_t BATCH2_SIZE = m_intensity;
    const uint32_t BWT_ALLOCATION_SIZE = BATCH2_SIZE * BWT_DATA_STRIDE;
    const uint32_t BATCH1_SIZE = (BWT_ALLOCATION_SIZE / STAGE1_DATA_STRIDE) & ~255U;

    return BATCH1_SIZE;
}


size_t xmrig::CudaAstroBWTRunner::processedHashes() const
{
    return CudaLib::deviceInt(m_ctx, CudaLib::DeviceAstroBWTProcessedHashes);
}
