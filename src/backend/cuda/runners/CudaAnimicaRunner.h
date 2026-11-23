/* XMRig
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 */

#ifndef XMRIG_CUDAANIMICARUNNER_H
#define XMRIG_CUDAANIMICARUNNER_H


#include "backend/cuda/runners/CudaBaseRunner.h"


namespace xmrig {


/**
 * CUDA runner for Animica hashshare PoW.
 *
 * Mirrors the CudaKawPow pattern: this class is just the dispatch
 * adapter — the actual SHA3-256 grid-stride kernel + nonce sweep lives
 * in the external xmrig-cuda plugin (a separate repository linked at
 * runtime via dlopen). The plugin must export the `animicaHash` symbol
 * with the signature documented in CudaLib.h; if it doesn't, CudaLib
 * gracefully reports the algorithm as unsupported and the OpenCL path
 * or CPU path picks up the work instead.
 *
 * The plugin-side kernel implementation tracks the same wire contract
 * the CPU and OpenCL runners use:
 *   - input is the 32-byte SHA3 prefix (prevhash) + the per-job 64-bit
 *     target (high 64 bits BE of the 256-bit target),
 *   - kernel computes sha3_256(prefix || nonce_le8) per thread,
 *   - writes a packed (nonce, digest) tuple back when digest <= target.
 */
class CudaAnimicaRunner : public CudaBaseRunner
{
public:
    CudaAnimicaRunner(size_t index, const CudaLaunchData &data);

protected:
    bool run(uint32_t startNonce, uint32_t *rescount, uint32_t *resnonce) override;
    bool set(const Job &job, uint8_t *blob) override;
    size_t processedHashes() const override { return intensity() - m_skippedHashes; }

private:
    uint8_t  *m_jobBlob       = nullptr;
    uint32_t  m_skippedHashes = 0;
    bool      m_warned        = false;     // throttle the "plugin missing" warning
};


} /* namespace xmrig */


#endif // XMRIG_CUDAANIMICARUNNER_H
