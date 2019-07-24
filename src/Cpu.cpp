/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 * Copyright 2018      Sebastian Stolzenberg <https://github.com/sebastianstolzenberg>
 *
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


#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>

#include <iostream>
#include <crypto/CryptoNight.h>
#include <crypto/Argon2.h>

#include "Cpu.h"
#include "CpuImpl.h"

CpuImpl& CpuImpl::instance()
{
    static CpuImpl cpu;
    return cpu;
}

CpuImpl::CpuImpl()
    : m_l2_exclusive(false)
    , m_brand{ 0 }
    , m_flags(0)
    , m_l2_cache(0)
    , m_l3_cache(0)
    , m_sockets(1)
    , m_totalCores(0)
    , m_totalThreads(0)
    , m_asmOptimization(AsmOptimization::ASM_OFF)
{
}

void CpuImpl::optimizeParameters(size_t& threadsCount, size_t& hashFactor,
                                 Options::Algo algo, PowVariant powVariant, size_t maxCpuUsage, bool safeMode)
{
    // limits hashfactor to maximum possible value defined by compiler flag
    hashFactor = std::min(hashFactor, (algo == Options::ALGO_CRYPTONIGHT_HEAVY || powVariant == POW_XFH) ? 3 : static_cast<size_t>(MAX_NUM_HASH_BLOCKS));

    if (!safeMode && threadsCount > 0 && hashFactor > 0)
    {
      // all parameters have been set manually and safe mode is off ... no optimization necessary
      return;
    }

    size_t cache = availableCache();
    size_t algoBlocks;
    switch (algo) {
        case Options::ALGO_CRYPTONIGHT_EXTREMELITE:
            algoBlocks = MEMORY_EXTREME_LITE/1024;
            break;
        case Options::ALGO_CRYPTONIGHT_ULTRALITE:
            algoBlocks = MEMORY_ULTRA_LITE/1024;
            break;
        case Options::ALGO_CRYPTONIGHT_SUPERLITE:
            algoBlocks = MEMORY_SUPER_LITE/1024;
            break;
        case Options::ALGO_CRYPTONIGHT_LITE:
            algoBlocks = MEMORY_LITE/1024;
            break;
        case Options::ALGO_CRYPTONIGHT_HEAVY:
            algoBlocks = MEMORY_HEAVY/1024;
            break;
        case Options::ALGO_ARGON2_256:
            algoBlocks = MEMORY_ARGON2_256/1024;
            break;
        case Options::ALGO_ARGON2_512:
            algoBlocks = MEMORY_ARGON2_512/1024;
            break;
        case Options::ALGO_CRYPTONIGHT:
        default:
            algoBlocks = MEMORY/1024;
            break;
    }

    size_t maximumReasonableFactor = std::max(cache / algoBlocks, static_cast<size_t>(1ul));
    size_t maximumReasonableThreadCount = std::min(maximumReasonableFactor, m_totalThreads);
    size_t maximumReasonableHashFactor = static_cast<size_t>(MAX_NUM_HASH_BLOCKS);

    if (algo == Options::ALGO_CRYPTONIGHT_HEAVY || powVariant == POW_XFH) {
        maximumReasonableHashFactor = 3;
    } else if (getCNBaseVariant(powVariant) == POW_V2 || getCNBaseVariant(powVariant) == POW_V4 || algo == Options::ALGO_CRYPTONIGHT_EXTREMELITE || algo == Options::ALGO_CRYPTONIGHT_ULTRALITE) {
        maximumReasonableHashFactor = 2;
    } else if (!Options::isCNAlgo(algo)) {
        maximumReasonableHashFactor = 1;
    }

    if (safeMode) {
        if (threadsCount > maximumReasonableThreadCount) {
            threadsCount = maximumReasonableThreadCount;
        }
        if (threadsCount > 0 && hashFactor > maximumReasonableFactor / threadsCount) {
            hashFactor = std::min(maximumReasonableFactor / threadsCount, maximumReasonableHashFactor);
            hashFactor  = std::max(hashFactor, static_cast<size_t>(1));
        }
    }

    if (threadsCount == 0) {
        if (hashFactor == 0) {
            threadsCount = maximumReasonableThreadCount;
        }
        else {
            threadsCount = std::min(maximumReasonableThreadCount,
                                    maximumReasonableFactor / hashFactor);
        }
        if (maxCpuUsage < 100)
        {
            threadsCount = std::min(threadsCount, m_totalThreads * maxCpuUsage / 100);
        }
        threadsCount = std::max(threadsCount, static_cast<size_t>(1));
    }

    if (hashFactor == 0) {
        hashFactor = std::min(maximumReasonableHashFactor, maximumReasonableFactor / threadsCount);
        hashFactor = std::max(hashFactor, static_cast<size_t>(1));
    }
}

bool CpuImpl::hasAES()
{
    return (m_flags & Cpu::AES) != 0;
}

bool CpuImpl::isX64()
{
    return (m_flags & Cpu::X86_64) != 0;
}

size_t CpuImpl::availableCache()
{
    size_t cache = 0;
    if (m_l3_cache) {
        cache = m_l2_exclusive ? (m_l2_cache + m_l3_cache) : m_l3_cache;
    }
    else {
        cache = m_l2_cache;
    }
    return cache;
}

void Cpu::init()
{
    CpuImpl::instance().init();
}

void Cpu::optimizeParameters(size_t& threadsCount, size_t& hashFactor, Options::Algo algo, PowVariant powVariant,
                               size_t maxCpuUsage, bool safeMode)
{
    CpuImpl::instance().optimizeParameters(threadsCount, hashFactor, algo, powVariant, maxCpuUsage, safeMode);
}

int Cpu::setThreadAffinity(size_t threadId, int64_t affinityMask)
{
    return CpuImpl::instance().setThreadAffinity(threadId, affinityMask);
}

bool Cpu::hasAES()
{
    return CpuImpl::instance().hasAES();
}

bool Cpu::isX64()
{
    return CpuImpl::instance().isX64();
}

const char* Cpu::brand()
{
    return CpuImpl::instance().brand();
}

size_t Cpu::cores()
{
    return CpuImpl::instance().cores();
}

size_t Cpu::l2()
{
    return CpuImpl::instance().l2();
}

size_t Cpu::l3()
{
    return CpuImpl::instance().l3();
}

size_t Cpu::sockets()
{
    return CpuImpl::instance().sockets();
}

size_t Cpu::threads()
{
    return CpuImpl::instance().threads();
}

size_t Cpu::availableCache()
{
    return CpuImpl::instance().availableCache();
}

int Cpu::getAssignedCpuId(size_t threadId, int64_t affinityMask)
{
    int cpuId = -1;

    Mem::ThreadBitSet threadAffinityMask = Mem::ThreadBitSet(affinityMask);
    size_t threadCount = 0;

    for (size_t i = 0; i < CpuImpl::instance().threads(); i++) {
        if (threadAffinityMask.test(i)) {
            if (threadCount == threadId) {
                cpuId = i;
                break;
            }

            threadCount++;
        }
    }

    return cpuId;
}

AsmOptimization Cpu::asmOptimization()
{
    return CpuImpl::instance().asmOptimization();
}
