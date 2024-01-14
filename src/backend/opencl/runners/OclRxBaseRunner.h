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

#ifndef XMRIG_OCLRXBASERUNNER_H
#define XMRIG_OCLRXBASERUNNER_H


#include "backend/opencl/runners/OclBaseRunner.h"
#include "base/tools/Buffer.h"


namespace xmrig {


class Blake2bHashRegistersKernel;
class Blake2bInitialHashKernel;
class Blake2bInitialHashDoubleKernel;
class FillAesKernel;
class FindSharesKernel;
class HashAesKernel;


class OclRxBaseRunner : public OclBaseRunner
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(OclRxBaseRunner)

    OclRxBaseRunner(size_t index, const OclLaunchData &data);
    ~OclRxBaseRunner() override;

protected:
    size_t bufferSize() const override;
    void build() override;
    void init() override;
    void run(uint32_t nonce, uint32_t *hashOutput) override;
    void set(const Job &job, uint8_t *blob) override;

protected:
    virtual void execute(uint32_t iteration) = 0;

    Blake2bHashRegistersKernel *m_blake2b_hash_registers_32       = nullptr;
    Blake2bHashRegistersKernel *m_blake2b_hash_registers_64       = nullptr;
    Blake2bInitialHashKernel *m_blake2b_initial_hash              = nullptr;
    Blake2bInitialHashDoubleKernel *m_blake2b_initial_hash_double = nullptr;
    Buffer m_seed;
    cl_mem m_dataset                                              = nullptr;
    cl_mem m_entropy                                              = nullptr;
    cl_mem m_hashes                                               = nullptr;
    cl_mem m_rounding                                             = nullptr;
    cl_mem m_scratchpads                                          = nullptr;
    FillAesKernel *m_fillAes1Rx4_scratchpad                       = nullptr;
    FillAesKernel *m_fillAes4Rx4_entropy                          = nullptr;
    FindSharesKernel *m_find_shares                               = nullptr;
    HashAesKernel *m_hashAes1Rx4                                  = nullptr;
    uint32_t m_gcn_version                                        = 12;
    uint32_t m_worksize                                           = 8;

    size_t m_jobSize                                              = 0;
};


} /* namespace xmrig */


#endif // XMRIG_OCLRXBASERUNNER_H
