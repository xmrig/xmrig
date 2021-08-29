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

#include "backend/opencl/runners/OclKawPowRunner.h"
#include "backend/common/Tags.h"
#include "3rdparty/libethash/ethash_internal.h"
#include "backend/opencl/kernels/kawpow/KawPow_CalculateDAGKernel.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/runners/tools/OclKawPow.h"
#include "backend/opencl/wrappers/OclError.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/io/log/Log.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/net/stratum/Job.h"
#include "base/tools/Chrono.h"
#include "crypto/common/VirtualMemory.h"
#include "crypto/kawpow/KPHash.h"


namespace xmrig {


constexpr size_t BLOB_SIZE = 40;


OclKawPowRunner::OclKawPowRunner(size_t index, const OclLaunchData &data) : OclBaseRunner(index, data)
{
    switch (data.thread.worksize())
    {
    case 64:
    case 128:
    case 256:
    case 512:
        m_workGroupSize = data.thread.worksize();
        break;
    }

    if (data.device.vendorId() == OclVendor::OCL_VENDOR_NVIDIA) {
        m_options += " -DPLATFORM=OPENCL_PLATFORM_NVIDIA";
        m_dagWorkGroupSize = 32;
    }
}


OclKawPowRunner::~OclKawPowRunner()
{
    OclLib::release(m_lightCache);
    OclLib::release(m_dag);

    delete m_calculateDagKernel;

    OclLib::release(m_controlQueue);
    OclLib::release(m_stop);

    OclKawPow::clear();
}


void OclKawPowRunner::run(uint32_t nonce, uint32_t *hashOutput)
{
    const size_t local_work_size = m_workGroupSize;
    const size_t global_work_offset = nonce;
    const size_t global_work_size = m_intensity - (m_intensity % m_workGroupSize);

    enqueueWriteBuffer(m_input, CL_FALSE, 0, BLOB_SIZE, m_blob);

    const uint32_t zero[2] = {};
    enqueueWriteBuffer(m_output, CL_FALSE, 0, sizeof(uint32_t), zero);
    enqueueWriteBuffer(m_stop, CL_FALSE, 0, sizeof(uint32_t) * 2, zero);

    m_skippedHashes = 0;

    const cl_int ret = OclLib::enqueueNDRangeKernel(m_queue, m_searchKernel, 1, &global_work_offset, &global_work_size, &local_work_size, 0, nullptr, nullptr);
    if (ret != CL_SUCCESS) {
        LOG_ERR("%s" RED(" error ") RED_BOLD("%s") RED(" when calling ") RED_BOLD("clEnqueueNDRangeKernel") RED(" for kernel ") RED_BOLD("progpow_search"),
            ocl_tag(), OclError::toString(ret));

        throw std::runtime_error(OclError::toString(ret));
    }

    uint32_t stop[2] = {};
    enqueueReadBuffer(m_stop, CL_FALSE, 0, sizeof(stop), stop);

    uint32_t output[16] = {};
    enqueueReadBuffer(m_output, CL_TRUE, 0, sizeof(output), output);

    m_skippedHashes = stop[1] * m_workGroupSize;

    if (output[0] > 15) {
        output[0] = 15;
    }

    hashOutput[0xFF] = output[0];
    memcpy(hashOutput, output + 1, output[0] * sizeof(uint32_t));
}


void OclKawPowRunner::set(const Job &job, uint8_t *blob)
{
    m_blockHeight = static_cast<uint32_t>(job.height());
    m_searchKernel = OclKawPow::get(*this, m_blockHeight, m_workGroupSize);

    const uint32_t epoch = m_blockHeight / KPHash::EPOCH_LENGTH;

    const uint64_t dag_size = KPCache::dag_size(epoch);
    if (dag_size > m_dagCapacity) {
        OclLib::release(m_dag);

        m_dagCapacity = VirtualMemory::align(dag_size, 16 * 1024 * 1024);
        m_dag = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE, m_dagCapacity);
    }

    if (epoch != m_epoch) {
        m_epoch = epoch;

        {
            std::lock_guard<std::mutex> lock(KPCache::s_cacheMutex);

            KPCache::s_cache.init(epoch);

            if (KPCache::s_cache.size() > m_lightCacheCapacity) {
                OclLib::release(m_lightCache);

                m_lightCacheCapacity = VirtualMemory::align(KPCache::s_cache.size());
                m_lightCache = OclLib::createBuffer(m_ctx, CL_MEM_READ_ONLY, m_lightCacheCapacity);
            }

            m_lightCacheSize = KPCache::s_cache.size();
            enqueueWriteBuffer(m_lightCache, CL_TRUE, 0, m_lightCacheSize, KPCache::s_cache.data());
        }

        const uint64_t start_ms = Chrono::steadyMSecs();

        const uint32_t dag_words = dag_size / sizeof(node);
        m_calculateDagKernel->setArgs(0, m_lightCache, m_dag, dag_words, m_lightCacheSize / sizeof(node));

        constexpr uint32_t N = 1 << 18;

        for (uint32_t start = 0; start < dag_words; start += N) {
            m_calculateDagKernel->setArg(0, sizeof(start), &start);
            m_calculateDagKernel->enqueue(m_queue, N, m_dagWorkGroupSize);
        }

        OclLib::finish(m_queue);

        LOG_INFO("%s " YELLOW("KawPow") " DAG for epoch " WHITE_BOLD("%u") " calculated " BLACK_BOLD("(%" PRIu64 "ms)"), Tags::opencl(), epoch, Chrono::steadyMSecs() - start_ms);
    }

    const uint64_t target = job.target();
    const uint32_t hack_false = 0;

    OclLib::setKernelArg(m_searchKernel, 0, sizeof(cl_mem), &m_dag);
    OclLib::setKernelArg(m_searchKernel, 1, sizeof(cl_mem), &m_input);
    OclLib::setKernelArg(m_searchKernel, 2, sizeof(target), &target);
    OclLib::setKernelArg(m_searchKernel, 3, sizeof(hack_false), &hack_false);
    OclLib::setKernelArg(m_searchKernel, 4, sizeof(cl_mem), &m_output);
    OclLib::setKernelArg(m_searchKernel, 5, sizeof(cl_mem), &m_stop);

    m_blob = blob;
    enqueueWriteBuffer(m_input, CL_TRUE, 0, BLOB_SIZE, m_blob);
}


void OclKawPowRunner::jobEarlyNotification(const Job&)
{
    const uint32_t one = 1;
    const cl_int ret = OclLib::enqueueWriteBuffer(m_controlQueue, m_stop, CL_TRUE, 0, sizeof(one), &one, 0, nullptr, nullptr);
    if (ret != CL_SUCCESS) {
        throw std::runtime_error(OclError::toString(ret));
    }
}


void xmrig::OclKawPowRunner::build()
{
    OclBaseRunner::build();

    m_calculateDagKernel = new KawPow_CalculateDAGKernel(m_program);
}


void xmrig::OclKawPowRunner::init()
{
    OclBaseRunner::init();

    m_controlQueue = OclLib::createCommandQueue(m_ctx, data().device.id());
    m_stop = OclLib::createBuffer(m_ctx, CL_MEM_READ_ONLY, sizeof(uint32_t) * 2);
}

} // namespace xmrig
