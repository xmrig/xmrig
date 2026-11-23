/* XMRig
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 */

#include "backend/cuda/runners/CudaAnimicaRunner.h"
#include "backend/cuda/CudaLaunchData.h"
#include "backend/cuda/wrappers/CudaLib.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/net/stratum/Job.h"


xmrig::CudaAnimicaRunner::CudaAnimicaRunner(size_t index, const CudaLaunchData &data) :
    CudaBaseRunner(index, data)
{
}


bool xmrig::CudaAnimicaRunner::run(uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce)
{
    if (!CudaLib::hasAnimicaSupport()) {
        if (!m_warned) {
            LOG_WARN("%s " YELLOW_BOLD("animica") YELLOW(" CUDA kernel not present in the loaded xmrig-cuda plugin; "
                "falling back to CPU/OpenCL. Build a plugin with WITH_ANIMICA=ON to enable GPU CUDA mining."),
                Tags::nvidia());
            m_warned = true;
        }
        *rescount = 0;
        return false;
    }
    // m_target inherited from CudaBaseRunner::set() holds the high-64
    // bits BE of the 256-bit target — same convention the OpenCL
    // runner uses. The plugin-side kernel expands it back to 32 bytes
    // BE for the per-nonce compare.
    return callWrapper(CudaLib::animicaHash(m_ctx, m_jobBlob, m_target,
                                             startNonce, rescount, resnonce,
                                             &m_skippedHashes));
}


bool xmrig::CudaAnimicaRunner::set(const Job &job, uint8_t *blob)
{
    if (!CudaBaseRunner::set(job, blob)) {
        return false;
    }
    m_jobBlob = blob;
    return true;
}
