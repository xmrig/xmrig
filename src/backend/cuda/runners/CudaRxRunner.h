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

#ifndef XMRIG_CUDARXRUNNER_H
#define XMRIG_CUDARXRUNNER_H


#include "backend/cuda/runners/CudaBaseRunner.h"
#include "base/tools/Buffer.h"


namespace xmrig {


class CudaRxRunner : public CudaBaseRunner
{
public:
    CudaRxRunner(size_t index, const CudaLaunchData &data);

protected:
    inline size_t intensity() const override { return m_intensity; }

    bool run(uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce) override;
    bool set(const Job &job, uint8_t *blob) override;

private:
    bool m_ready             = false;
    const bool m_datasetHost = false;
    Buffer m_seed;
    size_t m_intensity       = 0;
};


} /* namespace xmrig */


#endif // XMRIG_CUDARXRUNNER_H
