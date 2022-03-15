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

#ifndef XMRIG_OclAstroBWT_v2_Runner_H
#define XMRIG_OclAstroBWT_v2_Runner_H


#include "backend/opencl/runners/OclBaseRunner.h"

namespace xmrig {


class AstroBWT_v2_FindSharesKernel;
class AstroBWT_v2_BWT_FixOrderKernel;
class AstroBWT_v2_BWT_PreprocessKernel;
class AstroBWT_v2_Salsa20Kernel;
class AstroBWT_v2_SHA3InitialKernel;
class AstroBWT_v2_SHA3Kernel;


class OclAstroBWT_v2_Runner : public OclBaseRunner
{
public:
    static constexpr uint32_t BWT_DATA_SIZE = 9973;
    static constexpr uint32_t BWT_DATA_STRIDE = 10240;

    XMRIG_DISABLE_COPY_MOVE_DEFAULT(OclAstroBWT_v2_Runner)

    OclAstroBWT_v2_Runner(size_t index, const OclLaunchData &data);
    ~OclAstroBWT_v2_Runner() override;

    inline uint32_t roundSize() const override { return m_intensity; }
    inline uint32_t processedHashes() const override { return m_intensity; }

protected:
    size_t bufferSize() const override;
    void run(uint32_t nonce, uint32_t *hashOutput) override;
    void set(const Job &job, uint8_t *blob) override;
    void build() override;
    void init() override;

private:
    AstroBWT_v2_FindSharesKernel*       m_find_shares_kernel    = nullptr;
    AstroBWT_v2_BWT_FixOrderKernel*     m_bwt_fix_order_kernel  = nullptr;
    AstroBWT_v2_BWT_PreprocessKernel*   m_bwt_preprocess_kernel = nullptr;
    AstroBWT_v2_Salsa20Kernel*          m_salsa20_kernel        = nullptr;
    AstroBWT_v2_SHA3InitialKernel*      m_sha3_initial_kernel   = nullptr;
    AstroBWT_v2_SHA3Kernel*             m_sha3_kernel           = nullptr;

    cl_mem m_hashes         = nullptr;
    cl_mem m_data           = nullptr;
    cl_mem m_keys           = nullptr;
    cl_mem m_temp_storage   = nullptr;

    uint32_t m_workgroup_size = 0;
    uint32_t m_bwt_allocation_size = 0;
};


} /* namespace xmrig */


#endif // XMRIG_OclAstroBWT_v2_Runner_H
