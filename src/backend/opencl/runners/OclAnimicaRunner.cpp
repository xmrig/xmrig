/* XMRig
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 */

#include "backend/opencl/runners/OclAnimicaRunner.h"

#include <cstring>
#include <stdexcept>

#include "backend/opencl/cl/animica/animica_cl.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/wrappers/OclError.h"
#include "backend/opencl/wrappers/OclLib.h"
#include "base/net/stratum/Job.h"


namespace xmrig {


// Result record: [nonce_lo32, nonce_hi32, digest_u32 x 8] = 10 uint32 = 40 bytes.
constexpr uint32_t kResultStride = 10;


OclAnimicaRunner::OclAnimicaRunner(size_t index, const OclLaunchData &data) :
    OclBaseRunner(index, data)
{
    switch (data.thread.worksize()) {
    case 32: case 64: case 128: case 256: case 512:
        m_workGroupSize = data.thread.worksize();
        break;
    }
    // Nudge the OpenCL compiler toward dense scalar emission on NVIDIA
    // (no extra SIMD here — keccakf is dependency-chained per round so
    // wider vectors don't help much).
    if (data.device.vendorId() == OclVendor::OCL_VENDOR_NVIDIA) {
        m_options += " -DPLATFORM=OPENCL_PLATFORM_NVIDIA";
    }
}


OclAnimicaRunner::~OclAnimicaRunner()
{
    OclLib::release(m_searchKernel);
    OclLib::release(m_prefix);
    OclLib::release(m_target);
    OclLib::release(m_results);
}


size_t OclAnimicaRunner::bufferSize() const
{
    // OclBaseRunner reserves a single backing buffer of this size to
    // sub-allocate input/output ranges from. Animica's per-round
    // buffers (prefix, target, results) live in their own dedicated
    // allocations created in init(), so this is just the small
    // padding the base class still needs.
    return align(64);
}


void OclAnimicaRunner::init()
{
    // Point the base runner at the embedded SHA3-256 kernel source
    // before invoking the standard build/program-cache flow.
    m_source = animica_cl;

    OclBaseRunner::init();

    cl_int err = CL_SUCCESS;
    m_prefix = OclLib::createBuffer(m_ctx, CL_MEM_READ_ONLY,
                                     128, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OclAnimicaRunner: createBuffer(prefix) failed");
    }
    m_target = OclLib::createBuffer(m_ctx, CL_MEM_READ_ONLY,
                                     32, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OclAnimicaRunner: createBuffer(target) failed");
    }
    // Results: 1 count uint32 + kMaxResults * kResultStride uint32.
    m_results = OclLib::createBuffer(m_ctx, CL_MEM_READ_WRITE,
                                     (1 + kMaxResults * kResultStride) * sizeof(uint32_t),
                                     nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OclAnimicaRunner: createBuffer(results) failed");
    }
}


void OclAnimicaRunner::build()
{
    OclBaseRunner::build();
    cl_int err = CL_SUCCESS;
    m_searchKernel = OclLib::createKernel(m_program, "animica_search", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("OclAnimicaRunner: createKernel(animica_search) failed");
    }
}


void OclAnimicaRunner::set(const Job &job, uint8_t *blob)
{
    // The Stratum-level Animica job blob is layout
    //   blob[0..32) = prefix (prevhash bytes), blob[32..40) = nonce slot
    // Upload the prefix into the device buffer. The 8-byte nonce slot
    // in the host blob is unused on the GPU path — the kernel computes
    // each work-item's nonce from start_nonce + global_id(0).
    enqueueWriteBuffer(m_prefix, CL_TRUE, 0, kPrefixLen, blob);

    // Stretch job.target() (high 64 bits BE of the 256-bit target) into
    // a full 32-byte BE target buffer. The lower 192 bits are 0 — same
    // assumption the CPU path makes (animica share targets always have
    // leading zeroes; a tight target would otherwise be hashable in
    // microseconds and there'd be no need for vardiff).
    uint8_t target_be[32];
    std::memset(target_be, 0, sizeof(target_be));
    const uint64_t t = job.target();
    for (int i = 0; i < 8; ++i) {
        target_be[i] = static_cast<uint8_t>(t >> (8 * (7 - i)));
    }
    enqueueWriteBuffer(m_target, CL_TRUE, 0, sizeof(target_be), target_be);
}


void OclAnimicaRunner::run(uint32_t nonce, uint32_t /*nonce_offset*/, uint32_t *hashOutput)
{
    // Reset the host-visible share counter at results[0] every round.
    const uint32_t zero = 0;
    enqueueWriteBuffer(m_results, CL_FALSE, 0, sizeof(uint32_t), &zero);

    const uint64_t start_nonce_u64 = static_cast<uint64_t>(nonce);
    const uint32_t max_results     = kMaxResults;

    // Kernel arg layout matches animica.cl::animica_search.
    OclLib::setKernelArg(m_searchKernel, 0, sizeof(cl_mem),  &m_prefix);
    OclLib::setKernelArg(m_searchKernel, 1, sizeof(uint32_t), &m_prefixLen);
    OclLib::setKernelArg(m_searchKernel, 2, sizeof(uint64_t), &start_nonce_u64);
    OclLib::setKernelArg(m_searchKernel, 3, sizeof(cl_mem),  &m_target);
    OclLib::setKernelArg(m_searchKernel, 4, sizeof(cl_mem),  &m_results);
    OclLib::setKernelArg(m_searchKernel, 5, sizeof(uint32_t), &max_results);

    const size_t global_work_size = m_intensity - (m_intensity % m_workGroupSize);
    const size_t local_work_size  = m_workGroupSize;

    OclLib::enqueueNDRangeKernel(m_queue, m_searchKernel, 1,
                                  nullptr, &global_work_size, &local_work_size,
                                  0, nullptr, nullptr);

    // Read back the result count and any share records.
    uint32_t buf[1 + kMaxResults * kResultStride];
    enqueueReadBuffer(m_results, CL_TRUE, 0, sizeof(buf), buf);

    // Translate into the hashOutput layout the CPU/RX side already uses
    // (CN convention: 8 slots of 32-byte hashes laid out in m_hash).
    // The CpuWorker's submit path lifts the first 8 hashes' top-64-bits
    // and submits any that beat the target. For OpenCL Animica we
    // already know each entry is a winner (kernel did the compare), so
    // we just memcpy each digest into the corresponding slot.
    const uint32_t count = buf[0];
    const uint32_t emit  = count > 8 ? 8 : count;     // CpuWorker's m_hash is 8 * 32 bytes
    std::memset(hashOutput, 0, 8 * 32);
    for (uint32_t i = 0; i < emit; ++i) {
        std::memcpy(reinterpret_cast<uint8_t *>(hashOutput) + (i * 32),
                    &buf[1 + i * kResultStride + 2],
                    32);
    }
}


} /* namespace xmrig */
