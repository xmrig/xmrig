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
{
}

void CpuImpl::optimizeParameters(size_t& threadsCount, size_t& hashFactor,
                                 Options::Algo algo, size_t maxCpuUsage, bool safeMode)
{
    // limits hashfactor to maximum possible value defined by compiler flag
    hashFactor = std::min(hashFactor, algo == Options::ALGO_CRYPTONIGHT_HEAVY ? 3 : static_cast<size_t>(MAX_NUM_HASH_BLOCKS));

    if (!safeMode && threadsCount > 0 && hashFactor > 0)
    {
      // all parameters have been set manually and safe mode is off ... no optimization necessary
      return;
    }

    size_t cache = availableCache();
    size_t algoBlockSize;
    switch (algo) {
        case Options::ALGO_CRYPTONIGHT_LITE:
            algoBlockSize = 1024;
            break;
        case Options::ALGO_CRYPTONIGHT_HEAVY:
            algoBlockSize = 4096;
            break;
        case Options::ALGO_CRYPTONIGHT:
        default:
            algoBlockSize = 2048;
            break;
    }

    size_t maximumReasonableFactor = std::max(cache / algoBlockSize, static_cast<size_t>(1ul));
    size_t maximumReasonableThreadCount = std::min(maximumReasonableFactor, m_totalThreads);
    size_t maximumReasonableHashFactor = std::min(maximumReasonableFactor, algo == Options::ALGO_CRYPTONIGHT_HEAVY ? 3 : static_cast<size_t>(MAX_NUM_HASH_BLOCKS));

    if (safeMode) {
        if (threadsCount > maximumReasonableThreadCount) {
            threadsCount = maximumReasonableThreadCount;
        }
        if (hashFactor > maximumReasonableFactor / threadsCount) {
            hashFactor = std::min(maximumReasonableFactor / threadsCount, maximumReasonableHashFactor);
            hashFactor   = std::max(hashFactor, static_cast<size_t>(1));
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
        hashFactor   = std::max(hashFactor, static_cast<size_t>(1));
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

void Cpu::optimizeParameters(size_t& threadsCount, size_t& hashFactor, Options::Algo algo,
                               size_t maxCpuUsage, bool safeMode)
{
    CpuImpl::instance().optimizeParameters(threadsCount, hashFactor, algo, maxCpuUsage, safeMode);
}

void Cpu::setAffinity(int id, uint64_t mask)
{
    CpuImpl::instance().setAffinity(id, mask);
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
