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

#ifndef XMRIG_OCLKAWPOWRUNNER_H
#define XMRIG_OCLKAWPOWRUNNER_H


#include "backend/opencl/runners/OclBaseRunner.h"
#include "crypto/kawpow/KPCache.h"

#include <mutex>

namespace xmrig {


class KawPow_CalculateDAGKernel;


class OclKawPowRunner : public OclBaseRunner
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(OclKawPowRunner)

    OclKawPowRunner(size_t index, const OclLaunchData &data);
    ~OclKawPowRunner() override;

protected:
    void run(uint32_t nonce, uint32_t *hashOutput) override;
    void set(const Job &job, uint8_t *blob) override;
    void build() override;
    void init() override;
    void jobEarlyNotification(const Job& job) override;
    uint32_t processedHashes() const override { return m_intensity - m_skippedHashes; }

private:
    uint8_t* m_blob = nullptr;
    uint32_t m_skippedHashes = 0;

    uint32_t m_blockHeight = 0;
    uint32_t m_epoch = 0xFFFFFFFFUL;

    cl_mem m_lightCache = nullptr;
    size_t m_lightCacheSize = 0;
    size_t m_lightCacheCapacity = 0;

    cl_mem m_dag = nullptr;
    size_t m_dagCapacity = 0;

    KawPow_CalculateDAGKernel* m_calculateDagKernel = nullptr;

    cl_kernel m_searchKernel = nullptr;

    size_t m_workGroupSize = 256;
    size_t m_dagWorkGroupSize = 64;

    cl_command_queue m_controlQueue = nullptr;
    cl_mem m_stop = nullptr;
};


} /* namespace xmrig */


#endif // XMRIG_OCLKAWPOWRUNNER_H
