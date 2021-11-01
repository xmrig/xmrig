/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CUDAKAWPOWRUNNER_H
#define XMRIG_CUDAKAWPOWRUNNER_H


#include "backend/cuda/runners/CudaBaseRunner.h"


namespace xmrig {


class CudaKawPowRunner : public CudaBaseRunner
{
public:
    CudaKawPowRunner(size_t index, const CudaLaunchData &data);

protected:
    bool run(uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce) override;
    bool set(const Job &job, uint8_t *blob) override;
    size_t processedHashes() const override { return intensity() - m_skippedHashes; }
    void jobEarlyNotification(const Job&) override;

private:
    uint8_t* m_jobBlob = nullptr;
    uint32_t m_skippedHashes = 0;
};


} /* namespace xmrig */


#endif // XMRIG_CUDAKAWPOWRUNNER_H
