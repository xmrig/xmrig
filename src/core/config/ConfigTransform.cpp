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

#include "core/config/ConfigTransform.h"
#include "base/crypto/Algorithm.h"
#include "base/kernel/interfaces/IConfig.h"
#include "base/net/stratum/Pool.h"
#include "base/net/stratum/Pools.h"
#include "core/config/Config.h"
#include "crypto/cn/CnHash.h"


#ifdef XMRIG_ALGO_RANDOMX
#   include "crypto/rx/RxConfig.h"
#endif


#ifdef XMRIG_FEATURE_BENCHMARK
#   include "base/net/stratum/benchmark/BenchConfig.h"
#endif


namespace xmrig
{


static const char *kAffinity    = "affinity";
static const char *kAsterisk    = "*";
static const char *kEnabled     = "enabled";
static const char *kIntensity   = "intensity";
static const char *kThreads     = "threads";


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
        if (!doc.HasMember(CpuConfig::kField)) {
            doc.AddMember(StringRef(CpuConfig::kField), Value(kObjectType), allocator);
        }

        Value profile(kObjectType);
        profile.AddMember(StringRef(kIntensity), m_intensity, allocator);
        profile.AddMember(StringRef(kThreads),   m_threads, allocator);
        profile.AddMember(StringRef(kAffinity),  m_affinity, allocator);

#       ifdef XMRIG_ALGO_KAWPOW
        doc[CpuConfig::kField].AddMember(StringRef(Algorithm::kKAWPOW), false, doc.GetAllocator());
#       endif
        doc[CpuConfig::kField].AddMember(StringRef(kAsterisk), profile, doc.GetAllocator());
    }

#   ifdef XMRIG_FEATURE_OPENCL
    if (m_opencl) {
        set(doc, Config::kOcl, kEnabled, true);
    }
#   endif
}


void xmrig::ConfigTransform::transform(rapidjson::Document &doc, int key, const char *arg)
{
    BaseTransform::transform(doc, key, arg);

    switch (key) {
    case IConfig::AVKey:           /* --av */
    case IConfig::CPUPriorityKey:  /* --cpu-priority */
    case IConfig::ThreadsKey:      /* --threads */
    case IConfig::HugePageSizeKey: /* --hugepage-size */
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

    case IConfig::PauseOnBatteryKey: /* --pause-on-battery */
        return set(doc, Config::kPauseOnBattery, true);

    case IConfig::PauseOnActiveKey: /* --pause-on-active */
        return set(doc, Config::kPauseOnActive, static_cast<uint64_t>(strtol(arg, nullptr, 10)));

#   ifdef XMRIG_ALGO_ARGON2
    case IConfig::Argon2ImplKey: /* --argon2-impl */
        return set(doc, CpuConfig::kField, CpuConfig::kArgon2Impl, arg);
#   endif

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

    case IConfig::HugePagesJitKey: /* --huge-pages-jit */
        return set(doc, CpuConfig::kField, CpuConfig::kHugePagesJit, true);
#   endif

#   ifdef XMRIG_FEATURE_OPENCL
    case IConfig::OclKey: /* --opencl */
        m_opencl = true;
        break;

    case IConfig::OclCacheKey: /* --opencl-no-cache */
        return set(doc, Config::kOcl, "cache", false);

    case IConfig::OclLoaderKey: /* --opencl-loader */
        return set(doc, Config::kOcl, "loader", arg);

    case IConfig::OclDevicesKey: /* --opencl-devices */
        m_opencl = true;
        return set(doc, Config::kOcl, "devices-hint", arg);

    case IConfig::OclPlatformKey: /* --opencl-platform */
        if (strlen(arg) < 3) {
            return set(doc, Config::kOcl, "platform", static_cast<uint64_t>(strtol(arg, nullptr, 10)));
        }

        return set(doc, Config::kOcl, "platform", arg);
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    case IConfig::CudaKey: /* --cuda */
        return set(doc, Config::kCuda, kEnabled, true);

    case IConfig::CudaLoaderKey: /* --cuda-loader */
        return set(doc, Config::kCuda, "loader", arg);

    case IConfig::CudaDevicesKey: /* --cuda-devices */
        set(doc, Config::kCuda, kEnabled, true);
        return set(doc, Config::kCuda, "devices-hint", arg);

    case IConfig::CudaBFactorKey: /* --cuda-bfactor-hint */
        return set(doc, Config::kCuda, "bfactor-hint", static_cast<uint64_t>(strtol(arg, nullptr, 10)));

    case IConfig::CudaBSleepKey: /* --cuda-bsleep-hint */
        return set(doc, Config::kCuda, "bsleep-hint", static_cast<uint64_t>(strtol(arg, nullptr, 10)));
#   endif

#   ifdef XMRIG_FEATURE_NVML
    case IConfig::NvmlKey: /* --no-nvml */
        return set(doc, Config::kCuda, "nvml", false);
#   endif

#   if defined(XMRIG_FEATURE_NVML) || defined (XMRIG_FEATURE_ADL)
    case IConfig::HealthPrintTimeKey: /* --health-print-time */
        return set(doc, Config::kHealthPrintTime, static_cast<uint64_t>(strtol(arg, nullptr, 10)));
#   endif

#   ifdef XMRIG_FEATURE_DMI
    case IConfig::DmiKey: /* --no-dmi */
        return set(doc, Config::kDMI, false);
#   endif

#   ifdef XMRIG_FEATURE_BENCHMARK
    case IConfig::AlgorithmKey:     /* --algo */
    case IConfig::BenchKey:         /* --bench */
    case IConfig::StressKey:        /* --stress */
    case IConfig::BenchSubmitKey:   /* --submit */
    case IConfig::BenchVerifyKey:   /* --verify */
    case IConfig::BenchTokenKey:    /* --token */
    case IConfig::BenchSeedKey:     /* --seed */
    case IConfig::BenchHashKey:     /* --hash */
    case IConfig::UserKey:          /* --user */
    case IConfig::RotationKey:      /* --rotation */
        return transformBenchmark(doc, key, arg);
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

    case IConfig::HugePageSizeKey: /* --hugepage-size */
        return set(doc, CpuConfig::kField, CpuConfig::kHugePages, arg);

    default:
        break;
    }
}


#ifdef XMRIG_FEATURE_BENCHMARK
void xmrig::ConfigTransform::transformBenchmark(rapidjson::Document &doc, int key, const char *arg)
{
    set(doc, CpuConfig::kField, CpuConfig::kHugePagesJit, true);
    set(doc, CpuConfig::kField, CpuConfig::kPriority, 2);
    set(doc, CpuConfig::kField, CpuConfig::kYield, false);

    switch (key) {
    case IConfig::AlgorithmKey: /* --algo */
        return set(doc, BenchConfig::kBenchmark, BenchConfig::kAlgo, arg);

    case IConfig::BenchKey: /* --bench */
        return set(doc, BenchConfig::kBenchmark, BenchConfig::kSize, arg);

    case IConfig::StressKey: /* --stress */
        return add(doc, Pools::kPools, Pool::kUser, BenchConfig::kBenchmark);

    case IConfig::BenchSubmitKey: /* --submit */
        return set(doc, BenchConfig::kBenchmark, BenchConfig::kSubmit, true);

    case IConfig::BenchVerifyKey: /* --verify */
        return set(doc, BenchConfig::kBenchmark, BenchConfig::kVerify, arg);

    case IConfig::BenchTokenKey: /* --token */
        return set(doc, BenchConfig::kBenchmark, BenchConfig::kToken, arg);

    case IConfig::BenchSeedKey: /* --seed */
        return set(doc, BenchConfig::kBenchmark, BenchConfig::kSeed, arg);

    case IConfig::BenchHashKey: /* --hash */
        return set(doc, BenchConfig::kBenchmark, BenchConfig::kHash, arg);

    case IConfig::UserKey: /* --user */
        return set(doc, BenchConfig::kBenchmark, BenchConfig::kUser, arg);

    case IConfig::RotationKey: /* --rotation */
        return set(doc, BenchConfig::kBenchmark, BenchConfig::kRotation, arg);

    default:
        break;
    }
}
#endif

