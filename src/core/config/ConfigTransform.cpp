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


#include "base/kernel/interfaces/IConfig.h"
#include "backend/cpu/CpuConfig.h"
#include "base/net/stratum/Pool.h"
#include "base/net/stratum/Pools.h"
#include "core/config/ConfigTransform.h"
#include "crypto/cn/CnHash.h"


#ifdef XMRIG_ALGO_RANDOMX
#   include "crypto/rx/RxConfig.h"
#endif


namespace xmrig
{


static const char *kAffinity    = "affinity";
static const char *kAsterisk    = "*";
static const char *kEnabled     = "enabled";
static const char *kIntensity   = "intensity";
static const char *kThreads     = "threads";

#ifdef XMRIG_FEATURE_OPENCL
static const char *kOcl         = "opencl";
#endif

#ifdef XMRIG_FEATURE_CUDA
static const char *kCuda        = "cuda";
#endif


static inline uint64_t intensity(uint64_t av)
{
    switch (av) {
    case CnHash::AV_SINGLE:
    case CnHash::AV_SINGLE_SOFT:
        return 1;

    case CnHash::AV_DOUBLE_SOFT:
    case CnHash::AV_DOUBLE:
        return 2;

    case CnHash::AV_TRIPLE_SOFT:
    case CnHash::AV_TRIPLE:
        return 3;

    case CnHash::AV_QUAD_SOFT:
    case CnHash::AV_QUAD:
        return 4;

    case CnHash::AV_PENTA_SOFT:
    case CnHash::AV_PENTA:
        return 5;

    default:
        break;
    }

    return 1;
}


static inline bool isHwAes(uint64_t av)
{
    return av == CnHash::AV_SINGLE || av == CnHash::AV_DOUBLE || (av > CnHash::AV_DOUBLE_SOFT && av < CnHash::AV_TRIPLE_SOFT);
}


} // namespace xmrig


void xmrig::ConfigTransform::finalize(rapidjson::Document &doc)
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    BaseTransform::finalize(doc);

    if (m_threads) {
        doc.AddMember("version", 1, allocator);

        if (!doc.HasMember(CpuConfig::kField)) {
            doc.AddMember(StringRef(CpuConfig::kField), Value(kObjectType), allocator);
        }

        Value profile(kObjectType);
        profile.AddMember(StringRef(kIntensity), m_intensity, allocator);
        profile.AddMember(StringRef(kThreads),   m_threads, allocator);
        profile.AddMember(StringRef(kAffinity),  m_affinity, allocator);

        doc[CpuConfig::kField].AddMember(StringRef(kAsterisk), profile, doc.GetAllocator());
    }

#   ifdef XMRIG_FEATURE_OPENCL
    if (m_opencl) {
        set(doc, kOcl, kEnabled, true);
    }
#   endif
}


void xmrig::ConfigTransform::transform(rapidjson::Document &doc, int key, const char *arg)
{
    BaseTransform::transform(doc, key, arg);

    switch (key) {
    case IConfig::AVKey:          /* --av */
    case IConfig::CPUPriorityKey: /* --cpu-priority */
    case IConfig::ThreadsKey:     /* --threads */
        return transformUint64(doc, key, static_cast<uint64_t>(strtol(arg, nullptr, 10)));

    case IConfig::HugePagesKey: /* --no-huge-pages */
    case IConfig::CPUKey:       /* --no-cpu */
        return transformBoolean(doc, key, false);

    case IConfig::CPUAffinityKey: /* --cpu-affinity */
        {
            const char *p  = strstr(arg, "0x");
            return transformUint64(doc, key, p ? strtoull(p, nullptr, 16) : strtoull(arg, nullptr, 10));
        }

    case IConfig::CPUMaxThreadsKey: /* --cpu-max-threads-hint */
        return set(doc, CpuConfig::kField, CpuConfig::kMaxThreadsHint, static_cast<uint64_t>(strtol(arg, nullptr, 10)));

    case IConfig::MemoryPoolKey: /* --cpu-memory-pool */
        return set(doc, CpuConfig::kField, CpuConfig::kMemoryPool, static_cast<int64_t>(strtol(arg, nullptr, 10)));

    case IConfig::YieldKey: /* --cpu-no-yield */
        return set(doc, CpuConfig::kField, CpuConfig::kYield, false);

    case IConfig::Argon2ImplKey: /* --argon2-impl */
        return set(doc, CpuConfig::kField, CpuConfig::kArgon2Impl, arg);

#   ifdef XMRIG_FEATURE_ASM
    case IConfig::AssemblyKey: /* --asm */
        return set(doc, CpuConfig::kField, CpuConfig::kAsm, arg);
#   endif

#   ifdef XMRIG_ALGO_ASTROBWT
    case IConfig::AstroBWTMaxSizeKey: /* --astrobwt-max-size */
        return set(doc, CpuConfig::kField, CpuConfig::kAstroBWTMaxSize, static_cast<uint64_t>(strtol(arg, nullptr, 10)));

    case IConfig::AstroBWTAVX2Key: /* --astrobwt-avx2 */
        return set(doc, CpuConfig::kField, CpuConfig::kAstroBWTAVX2, true);
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    case IConfig::RandomXInitKey: /* --randomx-init */
        return set(doc, RxConfig::kField, RxConfig::kInit, static_cast<int64_t>(strtol(arg, nullptr, 10)));

#   ifdef XMRIG_FEATURE_HWLOC
    case IConfig::RandomXNumaKey: /* --randomx-no-numa */
        return set(doc, RxConfig::kField, RxConfig::kNUMA, false);
#   endif

    case IConfig::RandomXModeKey: /* --randomx-mode */
        return set(doc, RxConfig::kField, RxConfig::kMode, arg);

    case IConfig::RandomX1GbPagesKey: /* --randomx-1gb-pages */
        return set(doc, RxConfig::kField, RxConfig::kOneGbPages, true);

    case IConfig::RandomXWrmsrKey: /* --randomx-wrmsr */
        if (arg == nullptr) {
            return set(doc, RxConfig::kField, RxConfig::kWrmsr, true);
        }

        return set(doc, RxConfig::kField, RxConfig::kWrmsr, static_cast<int64_t>(strtol(arg, nullptr, 10)));

    case IConfig::RandomXRdmsrKey: /* --randomx-no-rdmsr */
        return set(doc, RxConfig::kField, RxConfig::kRdmsr, false);

    case IConfig::RandomXCacheQoSKey: /* --cache-qos */
        return set(doc, RxConfig::kField, RxConfig::kCacheQoS, true);
#   endif

#   ifdef XMRIG_FEATURE_OPENCL
    case IConfig::OclKey: /* --opencl */
        m_opencl = true;
        break;

    case IConfig::OclCacheKey: /* --opencl-no-cache */
        return set(doc, kOcl, "cache", false);

    case IConfig::OclLoaderKey: /* --opencl-loader */
        return set(doc, kOcl, "loader", arg);

    case IConfig::OclDevicesKey: /* --opencl-devices */
        m_opencl = true;
        return set(doc, kOcl, "devices-hint", arg);

    case IConfig::OclPlatformKey: /* --opencl-platform */
        if (strlen(arg) < 3) {
            return set(doc, kOcl, "platform", static_cast<uint64_t>(strtol(arg, nullptr, 10)));
        }

        return set(doc, kOcl, "platform", arg);
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    case IConfig::CudaKey: /* --cuda */
        return set(doc, kCuda, kEnabled, true);

    case IConfig::CudaLoaderKey: /* --cuda-loader */
        return set(doc, kCuda, "loader", arg);

    case IConfig::CudaDevicesKey: /* --cuda-devices */
        set(doc, kCuda, kEnabled, true);
        return set(doc, kCuda, "devices-hint", arg);

    case IConfig::CudaBFactorKey: /* --cuda-bfactor-hint */
        return set(doc, kCuda, "bfactor-hint", static_cast<uint64_t>(strtol(arg, nullptr, 10)));

    case IConfig::CudaBSleepKey: /* --cuda-bsleep-hint */
        return set(doc, kCuda, "bsleep-hint", static_cast<uint64_t>(strtol(arg, nullptr, 10)));
#   endif

#   ifdef XMRIG_FEATURE_NVML
    case IConfig::NvmlKey: /* --no-nvml */
        return set(doc, kCuda, "nvml", false);

    case IConfig::HealthPrintTimeKey: /* --health-print-time */
        return set(doc, "health-print-time", static_cast<uint64_t>(strtol(arg, nullptr, 10)));
#   endif

#   ifdef XMRIG_FEATURE_BENCHMARK
    case IConfig::StressKey: /* --stress */
    case IConfig::BenchKey:  /* --bench */
        set(doc, CpuConfig::kField, CpuConfig::kHugePagesJit, true);
        set(doc, CpuConfig::kField, CpuConfig::kPriority, 2);
        set(doc, CpuConfig::kField, CpuConfig::kYield, false);

        add(doc, Pools::kPools, Pool::kUser, Pool::kBenchmark);

        if (key == IConfig::BenchKey) {
            add(doc, Pools::kPools, Pool::kBenchmark, arg);
        }
        break;
#   endif

    default:
        break;
    }
}


void xmrig::ConfigTransform::transformBoolean(rapidjson::Document &doc, int key, bool enable)
{
    switch (key) {
    case IConfig::HugePagesKey: /* --no-huge-pages */
        return set(doc, CpuConfig::kField, CpuConfig::kHugePages, enable);

    case IConfig::CPUKey:       /* --no-cpu */
        return set(doc, CpuConfig::kField, kEnabled, enable);

    default:
        break;
    }
}


void xmrig::ConfigTransform::transformUint64(rapidjson::Document &doc, int key, uint64_t arg)
{
    using namespace rapidjson;

    switch (key) {
    case IConfig::CPUAffinityKey: /* --cpu-affinity */
        m_affinity = static_cast<int64_t>(arg);
        break;

    case IConfig::ThreadsKey: /* --threads */
        m_threads = arg;
        break;

    case IConfig::AVKey: /* --av */
        m_intensity = intensity(arg);
        set(doc, CpuConfig::kField, CpuConfig::kHwAes, isHwAes(arg));
        break;

    case IConfig::CPUPriorityKey: /* --cpu-priority */
        return set(doc, CpuConfig::kField, CpuConfig::kPriority, arg);

    default:
        break;
    }
}

