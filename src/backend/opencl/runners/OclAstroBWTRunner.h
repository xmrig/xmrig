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

#ifndef XMRIG_OCLASTROBWTRUNNER_H
#define XMRIG_OCLASTROBWTRUNNER_H


#include "backend/opencl/runners/OclBaseRunner.h"

namespace xmrig {


class AstroBWT_FilterKernel;
class AstroBWT_MainKernel;
class AstroBWT_PrepareBatch2Kernel;
class AstroBWT_Salsa20Kernel;
class AstroBWT_SHA3InitialKernel;
class AstroBWT_SHA3Kernel;
class AstroBWT_FindSharesKernel;


class OclAstroBWTRunner : public OclBaseRunner
{
public:
    static constexpr uint32_t BWT_DATA_MAX_SIZE = 600 * 1024 - 256;
    static constexpr uint32_t BWT_DATA_STRIDE = (BWT_DATA_MAX_SIZE + 256 + 255) & ~255U;

    XMRIG_DISABLE_COPY_MOVE_DEFAULT(OclAstroBWTRunner)

    OclAstroBWTRunner(size_t index, const OclLaunchData &data);
    ~OclAstroBWTRunner() override;

    inline uint32_t roundSize() const override { return static_cast<uint32_t>(m_batch_size1); }

    // ~0.5% of all hashes are invalid
    inline uint32_t processedHashes() const override { return static_cast<uint32_t>(m_processedHashes * 0.995); }

protected:
    size_t bufferSize() const override;
    void run(uint32_t nonce, uint32_t *hashOutput) override;
    void set(const Job &job, uint8_t *blob) override;
    void build() override;
    void init() override;

private:
    AstroBWT_SHA3InitialKernel*   m_sha3_initial_kernel   = nullptr;
    AstroBWT_SHA3Kernel*          m_sha3_kernel           = nullptr;
    AstroBWT_Salsa20Kernel*       m_salsa20_kernel        = nullptr;
    AstroBWT_MainKernel*          m_bwt_kernel            = nullptr;
    AstroBWT_FilterKernel*        m_filter_kernel         = nullptr;
    AstroBWT_PrepareBatch2Kernel* m_prepare_batch2_kernel = nullptr;
    AstroBWT_FindSharesKernel*    m_find_shares_kernel    = nullptr;


    cl_mem m_salsa20_keys                                 = nullptr;
    cl_mem m_bwt_data                                     = nullptr;
    cl_mem m_bwt_data_sizes                               = nullptr;
    cl_mem m_indices                                      = nullptr;
    cl_mem m_tmp_indices                                  = nullptr;
    cl_mem m_filtered_hashes                              = nullptr;

    uint32_t m_workgroup_size                             = 0;
    uint64_t m_bwt_allocation_size                        = 0;
    uint64_t m_batch_size1                                = 0;
    uint32_t m_processedHashes                            = 0;

    uint32_t* m_bwt_data_sizes_host                       = nullptr;
};


} /* namespace xmrig */


#endif // XMRIG_OCLASTROBWTRUNNER_H
