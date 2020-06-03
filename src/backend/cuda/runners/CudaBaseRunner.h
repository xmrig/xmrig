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

#ifndef XMRIG_CUDABASERUNNER_H
#define XMRIG_CUDABASERUNNER_H


#include "backend/cuda/interfaces/ICudaRunner.h"


using nvid_ctx = struct nvid_ctx;


namespace xmrig {


class CudaLaunchData;


class CudaBaseRunner : public ICudaRunner
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(CudaBaseRunner)

    CudaBaseRunner(size_t id, const CudaLaunchData &data);
    ~CudaBaseRunner() override;

protected:
    bool init() override;
    bool set(const Job &job, uint8_t *blob) override;
    size_t intensity() const override;
    size_t roundSize() const override { return intensity(); }
    size_t processedHashes() const override { return intensity(); }
    void jobEarlyNotification(const Job&) override {}

protected:
    bool callWrapper(bool result) const;

    const CudaLaunchData &m_data;
    const size_t m_threadId;
    nvid_ctx *m_ctx     = nullptr;
    uint64_t m_height   = 0;
    uint64_t m_target   = 0;
};


} /* namespace xmrig */


#endif // XMRIG_CUDABASERUNNER_H
