/* XMRig
 * Copyright (c) 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <support@xmrig.com>
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

#include <algorithm>
#include <array>
#include <cstring>
#include <thread>


#ifdef _MSC_VER
#   include <intrin.h>
#else
#   include <cpuid.h>
#endif


#include "crypto/cn/CryptoNight_monero.h"
#ifdef XMRIG_VAES
#   include "crypto/cn/CryptoNight_x86_vaes.h"
#endif


#include "backend/cpu/platform/BasicCpuInfo.h"
#include "3rdparty/rapidjson/document.h"
#include "crypto/common/Assembly.h"


#define VENDOR_ID                  (0)
#define PROCESSOR_INFO             (1)
#define EXTENDED_FEATURES          (7)
#define PROCESSOR_EXT_INFO         (0x80000001)
#define PROCESSOR_BRAND_STRING_1   (0x80000002)
#define PROCESSOR_BRAND_STRING_2   (0x80000003)
#define PROCESSOR_BRAND_STRING_3   (0x80000004)

#define EAX_Reg  (0)
#define EBX_Reg  (1)
#define ECX_Reg  (2)
#define EDX_Reg  (3)


namespace xmrig {


constexpr size_t kCpuFlagsSize                                  = 15;
static const std::array<const char *, kCpuFlagsSize> flagNames  = { "aes", "vaes", "avx", "avx2", "avx512f", "bmi2", "osxsave", "pdpe1gb", "sse2", "ssse3", "sse4.1", "xop", "popcnt", "cat_l3", "vm" };
static_assert(kCpuFlagsSize == ICpuInfo::FLAG_MAX, "kCpuFlagsSize and FLAG_MAX mismatch");


#ifdef XMRIG_FEATURE_MSR
constexpr size_t kMsrArraySize                                  = 6;
static const std::array<const char *, kMsrArraySize> msrNames   = { MSR_NAMES_LIST };
static_assert(kMsrArraySize == ICpuInfo::MSR_MOD_MAX, "kMsrArraySize and MSR_MOD_MAX mismatch");
#endif


static inline void cpuid(uint32_t level, int32_t output[4])
{
    memset(output, 0, sizeof(int32_t) * 4);

#   ifdef _MSC_VER
    __cpuidex(output, static_cast<int>(level), 0);
#   else
    __cpuid_count(level, 0, output[0], output[1], output[2], output[3]);
#   endif
}


static void cpu_brand_string(char out[64 + 6]) {
    int32_t cpu_info[4] = { 0 };
    char buf[64]        = { 0 };

    cpuid(VENDOR_ID, cpu_info);

    if (cpu_info[EAX_Reg] >= 4) {
        for (uint32_t i = 0; i < 4; i++) {
            cpuid(0x80000002 + i, cpu_info);
            memcpy(buf + (i * 16), cpu_info, sizeof(cpu_info));
        }
    }

    size_t pos        = 0;
    const size_t size = strlen(buf);

    for (size_t i = 0; i < size; ++i) {
        if (buf[i] == ' ' && ((pos > 0 && out[pos - 1] == ' ') || pos == 0)) {
            continue;
        }

        out[pos++] = buf[i];
    }

    if (pos > 0 && out[pos - 1] == ' ') {
        out[pos - 1] = '\0';
    }
}


static inline bool has_feature(uint32_t level, uint32_t reg, int32_t bit)
{
    int32_t cpu_info[4] = { 0 };
    cpuid(level, cpu_info);

    return (cpu_info[reg] & bit) != 0;
}


static inline int32_t get_masked(int32_t val, int32_t h, int32_t l)
{
    val &= (0x7FFFFFFF >> (31 - (h - l))) << l;
    return val >> l;
}


static inline uint64_t xgetbv()
{
#ifdef _MSC_VER
    return _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
#else
    uint32_t eax_reg = 0;
    uint32_t edx_reg = 0;
    __asm__ __volatile__("xgetbv": "=a"(eax_reg), "=d"(edx_reg) : "c"(0) : "cc");
    return (static_cast<uint64_t>(edx_reg) << 32) | eax_reg;
#endif
}

static inline bool has_xcr_avx()    { return (xgetbv() & 0x06) == 0x06; }
static inline bool has_xcr_avx512() { return (xgetbv() & 0xE6) == 0xE6; }
static inline bool has_osxsave()    { return has_feature(PROCESSOR_INFO,        ECX_Reg, 1 << 27); }
static inline bool has_aes_ni()     { return has_feature(PROCESSOR_INFO,        ECX_Reg, 1 << 25); }
static inline bool has_avx()        { return has_feature(PROCESSOR_INFO,        ECX_Reg, 1 << 28) && has_osxsave() && has_xcr_avx(); }
static inline bool has_avx2()       { return has_feature(EXTENDED_FEATURES,     EBX_Reg, 1 << 5) && has_osxsave() && has_xcr_avx(); }
static inline bool has_vaes()       { return has_feature(EXTENDED_FEATURES,     ECX_Reg, 1 << 9) && has_osxsave() && has_xcr_avx(); }
static inline bool has_avx512f()    { return has_feature(EXTENDED_FEATURES,     EBX_Reg, 1 << 16) && has_osxsave() && has_xcr_avx512(); }
static inline bool has_bmi2()       { return has_feature(EXTENDED_FEATURES,     EBX_Reg, 1 << 8); }
static inline bool has_pdpe1gb()    { return has_feature(PROCESSOR_EXT_INFO,    EDX_Reg, 1 << 26); }
static inline bool has_sse2()       { return has_feature(PROCESSOR_INFO,        EDX_Reg, 1 << 26); }
static inline bool has_ssse3()      { return has_feature(PROCESSOR_INFO,        ECX_Reg, 1 << 9); }
static inline bool has_sse41()      { return has_feature(PROCESSOR_INFO,        ECX_Reg, 1 << 19); }
static inline bool has_xop()        { return has_feature(0x80000001,            ECX_Reg, 1 << 11); }
static inline bool has_popcnt()     { return has_feature(PROCESSOR_INFO,        ECX_Reg, 1 << 23); }
static inline bool has_cat_l3()     { return has_feature(EXTENDED_FEATURES,     EBX_Reg, 1 << 15) && has_feature(0x10, EBX_Reg, 1 << 1); }
static inline bool is_vm()          { return has_feature(PROCESSOR_INFO,        ECX_Reg, 1 << 31); }


} // namespace xmrig


#ifdef XMRIG_ALGO_ARGON2
extern "C" {


int cpu_flags_has_avx2()    { return xmrig::has_avx2(); }
int cpu_flags_has_avx512f() { return xmrig::has_avx512f(); }
int cpu_flags_has_sse2()    { return xmrig::has_sse2(); }
int cpu_flags_has_ssse3()   { return xmrig::has_ssse3(); }
int cpu_flags_has_xop()     { return xmrig::has_xop(); }


}
#endif


xmrig::BasicCpuInfo::BasicCpuInfo() :
    m_threads(std::thread::hardware_concurrency())
{
    cpu_brand_string(m_brand);

    m_flags.set(FLAG_AES,     has_aes_ni());
    m_flags.set(FLAG_AVX,     has_avx());
    m_flags.set(FLAG_AVX2,    has_avx2());
    m_flags.set(FLAG_VAES,    has_vaes());
    m_flags.set(FLAG_AVX512F, has_avx512f());
    m_flags.set(FLAG_BMI2,    has_bmi2());
    m_flags.set(FLAG_OSXSAVE, has_osxsave());
    m_flags.set(FLAG_PDPE1GB, has_pdpe1gb());
    m_flags.set(FLAG_SSE2,    has_sse2());
    m_flags.set(FLAG_SSSE3,   has_ssse3());
    m_flags.set(FLAG_SSE41,   has_sse41());
    m_flags.set(FLAG_XOP,     has_xop());
    m_flags.set(FLAG_POPCNT,  has_popcnt());
    m_flags.set(FLAG_CAT_L3,  has_cat_l3());
    m_flags.set(FLAG_VM,      is_vm());

    m_units.resize(m_threads);
    for (int32_t i = 0; i < static_cast<int32_t>(m_threads); ++i) {
        m_units[i] = i;
    }

#   ifdef XMRIG_FEATURE_ASM
    if (m_flags.test(FLAG_AES)) {
        char vendor[13] = { 0 };
        int32_t data[4] = { 0 };

        cpuid(VENDOR_ID, data);

        memcpy(vendor + 0, &data[1], 4);
        memcpy(vendor + 4, &data[3], 4);
        memcpy(vendor + 8, &data[2], 4);

        cpuid(PROCESSOR_INFO, data);

        m_procInfo = data[EAX_Reg];
        m_family = get_masked(m_procInfo, 12, 8) + get_masked(m_procInfo, 28, 20);
        m_model = (get_masked(m_procInfo, 20, 16) << 4) | get_masked(m_procInfo, 8, 4);
        m_stepping = get_masked(m_procInfo, 4, 0);

        if (memcmp(vendor, "AuthenticAMD", 12) == 0) {
            m_vendor = VENDOR_AMD;

            if (m_family >= 0x17) {
                m_assembly = Assembly::RYZEN;

                switch (m_family) {
                case 0x17:
                    m_msrMod = MSR_MOD_RYZEN_17H;
                    switch (m_model) {
                    case 1:
                    case 17:
                    case 32:
                        m_arch = ARCH_ZEN;
                        break;
                    case 8:
                    case 24:
                        m_arch = ARCH_ZEN_PLUS;
                        break;
                    case 49:
                    case 96:
                    case 113:
                    case 144:
                        m_arch = ARCH_ZEN2;
                        break;
                    }
                    break;

                case 0x19:
                    if (m_model == 0x61) {
                        m_arch = ARCH_ZEN4;
                        m_msrMod = MSR_MOD_RYZEN_19H_ZEN4;
                    }
                    else {
                        m_arch = ARCH_ZEN3;
                        m_msrMod = MSR_MOD_RYZEN_19H;
                    }
                    break;

                default:
                    m_msrMod = MSR_MOD_NONE;
                    break;
                }
            }
            else {
                m_assembly = Assembly::BULLDOZER;
            }
        }
        else if (memcmp(vendor, "GenuineIntel", 12) == 0) {
            m_vendor   = VENDOR_INTEL;
            m_assembly = Assembly::INTEL;
            m_msrMod   = MSR_MOD_INTEL;

            struct
            {
                unsigned int stepping : 4;
                unsigned int model : 4;
                unsigned int family : 4;
                unsigned int processor_type : 2;
                unsigned int reserved1 : 2;
                unsigned int ext_model : 4;
                unsigned int ext_family : 8;
                unsigned int reserved2 : 4;
            } processor_info;

            memcpy(&processor_info, data, sizeof(processor_info));

            // Intel JCC erratum mitigation
            if (processor_info.family == 6) {
                const uint32_t model = processor_info.model | (processor_info.ext_model << 4);
                const uint32_t stepping = processor_info.stepping;

                // Affected CPU models and stepping numbers are taken from https://www.intel.com/content/dam/support/us/en/documents/processors/mitigations-jump-conditional-code-erratum.pdf
                m_jccErratum =
                    ((model == 0x4E) && (stepping == 0x3)) ||
                    ((model == 0x55) && ((stepping == 0x4) || (stepping == 0x7))) ||
                    ((model == 0x5E) && (stepping == 0x3)) ||
                    ((model == 0x8E) && (stepping >= 0x9) && (stepping <= 0xC)) ||
                    ((model == 0x9E) && (stepping >= 0x9) && (stepping <= 0xD)) ||
                    ((model == 0xA6) && (stepping == 0x0)) ||
                    ((model == 0xAE) && (stepping == 0xA));
            }
        }
    }
#   endif

    cn_sse41_enabled = has(FLAG_SSE41);
    cn_vaes_enabled = has(FLAG_VAES);
}


const char *xmrig::BasicCpuInfo::backend() const
{
    return "basic/1";
}


xmrig::CpuThreads xmrig::BasicCpuInfo::threads(const Algorithm &algorithm, uint32_t limit) const
{
    const uint32_t count = std::thread::hardware_concurrency();
    const uint32_t count_limit  = std::max(static_cast<uint32_t>(count * (limit / 100.0f)), 1U);
    const uint32_t count_limit2 = std::max(static_cast<uint32_t>(count / 2), count_limit);
    const uint32_t count_limit4 = std::max(static_cast<uint32_t>(count / 4), count_limit);

    if (count == 1) {
        return 1;
    }

    const auto f = algorithm.family();

#   ifdef XMRIG_ALGO_CN_LITE
    if (f == Algorithm::CN_LITE) {
        return CpuThreads(count_limit, 1);
    }
#   endif

#   ifdef XMRIG_ALGO_CN_PICO
    if (f == Algorithm::CN_PICO) {
        return CpuThreads(count_limit, 2);
    }
#   endif

#   ifdef XMRIG_ALGO_CN_FEMTO
    if (f == Algorithm::CN_FEMTO) {
        return CpuThreads(count, 2);
    }
#   endif

#   ifdef XMRIG_ALGO_CN_HEAVY
    if (f == Algorithm::CN_HEAVY) {
        return CpuThreads(count_limit4, 1);
    }
#   endif

#   ifdef XMRIG_ALGO_CN_GPU
    if (algorithm == Algorithm::CN_GPU) {
        return count_limit;
    }
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    if (f == Algorithm::RANDOM_X) {
        if (algorithm == Algorithm::RX_WOW) {
            return count_limit;
        }

        if (algorithm == Algorithm::RX_XLA) {
            CpuThreads threads;
            for (size_t i = 0; i < count_limit2; ++i) {
                threads.add(i, 0);
            }
            return threads;
        }

        return count_limit2;
    }
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    if (f == Algorithm::ARGON2) {
        return count_limit;
    }
#   endif

#   ifdef XMRIG_ALGO_GHOSTRIDER
    switch (algorithm.id()) {
        case Algorithm::GHOSTRIDER_RTM: return CpuThreads(std::max<size_t>(count_limit2, 1), 8);
        case Algorithm::FLEX_KCN:       return CpuThreads(std::max<size_t>(count_limit2, 1), 1);
        default:
            break;
    }
#   endif

    return CpuThreads(count_limit2, 1);
}


rapidjson::Value xmrig::BasicCpuInfo::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;
    auto &allocator = doc.GetAllocator();

    Value out(kObjectType);

    out.AddMember("brand",      StringRef(brand()), allocator);
    out.AddMember("family",     m_family, allocator);
    out.AddMember("model",      m_model, allocator);
    out.AddMember("stepping",   m_stepping, allocator);
    out.AddMember("proc_info",  m_procInfo, allocator);
    out.AddMember("aes",        hasAES(), allocator);
    out.AddMember("avx2",       hasAVX2(), allocator);
    out.AddMember("x64",        is64bit(), allocator); // DEPRECATED will be removed in the next major release.
    out.AddMember("64_bit",     is64bit(), allocator);
    out.AddMember("l2",         static_cast<uint64_t>(L2()), allocator);
    out.AddMember("l3",         static_cast<uint64_t>(L3()), allocator);
    out.AddMember("cores",      static_cast<uint64_t>(cores()), allocator);
    out.AddMember("threads",    static_cast<uint64_t>(threads()), allocator);
    out.AddMember("packages",   static_cast<uint64_t>(packages()), allocator);
    out.AddMember("nodes",      static_cast<uint64_t>(nodes()), allocator);
    out.AddMember("backend",    StringRef(backend()), allocator);

#   ifdef XMRIG_FEATURE_MSR
    out.AddMember("msr",        StringRef(msrNames[msrMod()]), allocator);
#   else
    out.AddMember("msr",        "none", allocator);
#   endif

#   ifdef XMRIG_FEATURE_ASM
    out.AddMember("assembly",   StringRef(Assembly(assembly()).toString()), allocator);
#   else
    out.AddMember("assembly",   "none", allocator);
#   endif

#   if defined(__x86_64__) || defined(_M_AMD64)
    out.AddMember("arch", "x86_64", allocator);
#   else
    out.AddMember("arch", "x86", allocator);
#   endif

    Value flags(kArrayType);

    for (size_t i = 0; i < flagNames.size(); ++i) {
        if (m_flags.test(i)) {
            flags.PushBack(StringRef(flagNames[i]), allocator);
        }
    }

    out.AddMember("flags", flags, allocator);

    return out;
}
