/* XMRig
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 */

#ifndef XMRIG_OCLANIMICARUNNER_H
#define XMRIG_OCLANIMICARUNNER_H


#include "backend/opencl/runners/OclBaseRunner.h"


namespace xmrig {


/**
 * OpenCL runner for Animica hashshare PoW.
 *
 * One work-item per nonce. The kernel (src/backend/opencl/cl/animica/
 * animica.cl) hashes `prefix || nonce_le8` with SHA3-256, compares the
 * 32-byte big-endian digest against the per-job target, and atomic-
 * reserves a slot in the host-visible results buffer on a hit.
 *
 * Buffer layout:
 *   m_input    — `[prefix_bytes (32) || nonce_slot (8) || target_be (32)]`
 *                (sized to a flat 80-byte page; only the first 32 + 32
 *                bytes change per job — the worker re-uploads only when
 *                set() is called.)
 *   m_output   — `[count_u32, nonce_lo32, nonce_hi32, digest_u32 x 8, ...]`
 *                Caller reads count, then `count` share records (40 B
 *                each). Capacity is bounded by `kMaxResults`.
 *
 * Intensity ("global work size") defaults to `data().thread.intensity()`
 * which the OpenCL config layer auto-tunes per device. There's no
 * dataset to upload (cf. KawPow's DAG) so init() is one-shot and
 * build() is unconditional.
 */
class OclAnimicaRunner : public OclBaseRunner
{
public:
    XMRIG_DISABLE_COPY_MOVE_DEFAULT(OclAnimicaRunner)

    OclAnimicaRunner(size_t index, const OclLaunchData &data);
    ~OclAnimicaRunner() override;

protected:
    void run(uint32_t nonce, uint32_t nonce_offset, uint32_t *hashOutput) override;
    void set(const Job &job, uint8_t *blob) override;
    void build() override;
    void init() override;
    size_t bufferSize() const override;

private:
    static constexpr uint32_t kMaxResults = 32;     // shares per round before we drop excess
    static constexpr size_t   kPrefixLen  = 32;     // Stratum prevhash size in bytes

    cl_kernel m_searchKernel = nullptr;
    cl_mem    m_prefix       = nullptr;     // __constant uchar* prefix_bytes
    cl_mem    m_target       = nullptr;     // __constant uchar* target_be_32B
    cl_mem    m_results      = nullptr;     // __global   uint*  [count, (nonce_lo, nonce_hi, digest_u32x8)...]
    uint32_t  m_prefixLen    = kPrefixLen;
    size_t    m_workGroupSize = 256;
};


} /* namespace xmrig */


#endif // XMRIG_OCLANIMICARUNNER_H
