/* XMRig
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "crypto/cn/CnHash.h"
#include "backend/cpu/Cpu.h"
#include "base/tools/cryptonote/umul128.h"
#include "crypto/common/VirtualMemory.h"


#if defined(XMRIG_ARM)
#   include "crypto/cn/CryptoNight_arm.h"
#else
#   include "crypto/cn/CryptoNight_x86.h"
#endif


#ifdef XMRIG_ALGO_ARGON2
#   include "crypto/argon2/Hash.h"
#endif


#ifdef XMRIG_ALGO_ASTROBWT
#   include "crypto/astrobwt/AstroBWT.h"
#endif


#define ADD_FN(algo) do {                                                                            \
        m_map[algo] = new cn_hash_fun_array{};                                                       \
        m_map[algo]->data[AV_SINGLE][Assembly::NONE]      = cryptonight_single_hash<algo, false, 0>; \
        m_map[algo]->data[AV_SINGLE_SOFT][Assembly::NONE] = cryptonight_single_hash<algo, true,  0>; \
        m_map[algo]->data[AV_DOUBLE][Assembly::NONE]      = cryptonight_double_hash<algo, false>;    \
        m_map[algo]->data[AV_DOUBLE_SOFT][Assembly::NONE] = cryptonight_double_hash<algo, true>;     \
        m_map[algo]->data[AV_TRIPLE][Assembly::NONE]      = cryptonight_triple_hash<algo, false>;    \
        m_map[algo]->data[AV_TRIPLE_SOFT][Assembly::NONE] = cryptonight_triple_hash<algo, true>;     \
        m_map[algo]->data[AV_QUAD][Assembly::NONE]        = cryptonight_quad_hash<algo,   false>;    \
        m_map[algo]->data[AV_QUAD_SOFT][Assembly::NONE]   = cryptonight_quad_hash<algo,   true>;     \
        m_map[algo]->data[AV_PENTA][Assembly::NONE]       = cryptonight_penta_hash<algo,  false>;    \
        m_map[algo]->data[AV_PENTA_SOFT][Assembly::NONE]  = cryptonight_penta_hash<algo,  true>;     \
    } while (0)


#ifdef XMRIG_FEATURE_ASM
#   define ADD_FN_ASM(algo) do {                                                                                    \
        m_map[algo]->data[AV_SINGLE][Assembly::INTEL]     = cryptonight_single_hash_asm<algo, Assembly::INTEL>;     \
        m_map[algo]->data[AV_SINGLE][Assembly::RYZEN]     = cryptonight_single_hash_asm<algo, Assembly::RYZEN>;     \
        m_map[algo]->data[AV_SINGLE][Assembly::BULLDOZER] = cryptonight_single_hash_asm<algo, Assembly::BULLDOZER>; \
        m_map[algo]->data[AV_DOUBLE][Assembly::INTEL]     = cryptonight_double_hash_asm<algo, Assembly::INTEL>;     \
        m_map[algo]->data[AV_DOUBLE][Assembly::RYZEN]     = cryptonight_double_hash_asm<algo, Assembly::RYZEN>;     \
        m_map[algo]->data[AV_DOUBLE][Assembly::BULLDOZER] = cryptonight_double_hash_asm<algo, Assembly::BULLDOZER>; \
    } while (0)


namespace xmrig {


cn_mainloop_fun        cn_half_mainloop_ivybridge_asm             = nullptr;
cn_mainloop_fun        cn_half_mainloop_ryzen_asm                 = nullptr;
cn_mainloop_fun        cn_half_mainloop_bulldozer_asm             = nullptr;
cn_mainloop_fun        cn_half_double_mainloop_sandybridge_asm    = nullptr;

cn_mainloop_fun        cn_trtl_mainloop_ivybridge_asm             = nullptr;
cn_mainloop_fun        cn_trtl_mainloop_ryzen_asm                 = nullptr;
cn_mainloop_fun        cn_trtl_mainloop_bulldozer_asm             = nullptr;
cn_mainloop_fun        cn_trtl_double_mainloop_sandybridge_asm    = nullptr;

cn_mainloop_fun        cn_tlo_mainloop_ivybridge_asm              = nullptr;
cn_mainloop_fun        cn_tlo_mainloop_ryzen_asm                  = nullptr;
cn_mainloop_fun        cn_tlo_mainloop_bulldozer_asm              = nullptr;
cn_mainloop_fun        cn_tlo_double_mainloop_sandybridge_asm     = nullptr;

cn_mainloop_fun        cn_zls_mainloop_ivybridge_asm              = nullptr;
cn_mainloop_fun        cn_zls_mainloop_ryzen_asm                  = nullptr;
cn_mainloop_fun        cn_zls_mainloop_bulldozer_asm              = nullptr;
cn_mainloop_fun        cn_zls_double_mainloop_sandybridge_asm     = nullptr;

cn_mainloop_fun        cn_double_mainloop_ivybridge_asm           = nullptr;
cn_mainloop_fun        cn_double_mainloop_ryzen_asm               = nullptr;
cn_mainloop_fun        cn_double_mainloop_bulldozer_asm           = nullptr;
cn_mainloop_fun        cn_double_double_mainloop_sandybridge_asm  = nullptr;

cn_mainloop_fun        cn_upx2_mainloop_asm                       = nullptr;
cn_mainloop_fun        cn_upx2_double_mainloop_asm                = nullptr;


template<Algorithm::Id SOURCE_ALGO = Algorithm::CN_2, typename T, typename U>
static void patchCode(T dst, U src, const uint32_t iterations, const uint32_t mask = CnAlgo<Algorithm::CN_HALF>().mask())
{
    auto p = reinterpret_cast<const uint8_t*>(src);

    // Workaround for Visual Studio placing trampoline in debug builds.
#   if defined(_MSC_VER)
    if (p[0] == 0xE9) {
        p += *(int32_t*)(p + 1) + 5;
    }
#   endif

    size_t size = 0;
    while (*(uint32_t*)(p + size) != 0xDEADC0DE) {
        ++size;
    }

    size += sizeof(uint32_t);

    memcpy((void*) dst, (const void*) src, size);

    auto patched_data = reinterpret_cast<uint8_t*>(dst);
    for (size_t i = 0; i + sizeof(uint32_t) <= size; ++i) {
        switch (*(uint32_t*)(patched_data + i)) {
        case CnAlgo<SOURCE_ALGO>().iterations():
            *(uint32_t*)(patched_data + i) = iterations;
            break;

        case CnAlgo<SOURCE_ALGO>().mask():
            *(uint32_t*)(patched_data + i) = mask;
            break;
        }
    }
}


static void patchAsmVariants()
{
    const int allocation_size = 131072;
    auto base = static_cast<uint8_t *>(VirtualMemory::allocateExecutableMemory(allocation_size, false));

    cn_half_mainloop_ivybridge_asm              = reinterpret_cast<cn_mainloop_fun>         (base + 0x0000);
    cn_half_mainloop_ryzen_asm                  = reinterpret_cast<cn_mainloop_fun>         (base + 0x1000);
    cn_half_mainloop_bulldozer_asm              = reinterpret_cast<cn_mainloop_fun>         (base + 0x2000);
    cn_half_double_mainloop_sandybridge_asm     = reinterpret_cast<cn_mainloop_fun>         (base + 0x3000);

#   ifdef XMRIG_ALGO_CN_PICO
    cn_trtl_mainloop_ivybridge_asm              = reinterpret_cast<cn_mainloop_fun>         (base + 0x4000);
    cn_trtl_mainloop_ryzen_asm                  = reinterpret_cast<cn_mainloop_fun>         (base + 0x5000);
    cn_trtl_mainloop_bulldozer_asm              = reinterpret_cast<cn_mainloop_fun>         (base + 0x6000);
    cn_trtl_double_mainloop_sandybridge_asm     = reinterpret_cast<cn_mainloop_fun>         (base + 0x7000);
#   endif

    cn_zls_mainloop_ivybridge_asm               = reinterpret_cast<cn_mainloop_fun>         (base + 0x8000);
    cn_zls_mainloop_ryzen_asm                   = reinterpret_cast<cn_mainloop_fun>         (base + 0x9000);
    cn_zls_mainloop_bulldozer_asm               = reinterpret_cast<cn_mainloop_fun>         (base + 0xA000);
    cn_zls_double_mainloop_sandybridge_asm      = reinterpret_cast<cn_mainloop_fun>         (base + 0xB000);

    cn_double_mainloop_ivybridge_asm            = reinterpret_cast<cn_mainloop_fun>         (base + 0xC000);
    cn_double_mainloop_ryzen_asm                = reinterpret_cast<cn_mainloop_fun>         (base + 0xD000);
    cn_double_mainloop_bulldozer_asm            = reinterpret_cast<cn_mainloop_fun>         (base + 0xE000);
    cn_double_double_mainloop_sandybridge_asm   = reinterpret_cast<cn_mainloop_fun>         (base + 0xF000);

#   ifdef XMRIG_ALGO_CN_PICO
    cn_tlo_mainloop_ivybridge_asm               = reinterpret_cast<cn_mainloop_fun>         (base + 0x10000);
    cn_tlo_mainloop_ryzen_asm                   = reinterpret_cast<cn_mainloop_fun>         (base + 0x11000);
    cn_tlo_mainloop_bulldozer_asm               = reinterpret_cast<cn_mainloop_fun>         (base + 0x12000);
    cn_tlo_double_mainloop_sandybridge_asm      = reinterpret_cast<cn_mainloop_fun>         (base + 0x13000);
#   endif

#   ifdef XMRIG_ALGO_CN_FEMTO
    cn_upx2_mainloop_asm                        = reinterpret_cast<cn_mainloop_fun>         (base + 0x14000);
    cn_upx2_double_mainloop_asm                 = reinterpret_cast<cn_mainloop_fun>         (base + 0x15000);
#   endif

    {
        constexpr uint32_t ITER = CnAlgo<Algorithm::CN_HALF>().iterations();

        patchCode(cn_half_mainloop_ivybridge_asm,            cnv2_mainloop_ivybridge_asm,           ITER);
        patchCode(cn_half_mainloop_ryzen_asm,                cnv2_mainloop_ryzen_asm,               ITER);
        patchCode(cn_half_mainloop_bulldozer_asm,            cnv2_mainloop_bulldozer_asm,           ITER);
        patchCode(cn_half_double_mainloop_sandybridge_asm,   cnv2_double_mainloop_sandybridge_asm,  ITER);
    }

#   ifdef XMRIG_ALGO_CN_PICO
    {
        constexpr uint32_t ITER = CnAlgo<Algorithm::CN_PICO_0>().iterations();
        constexpr uint32_t MASK = CnAlgo<Algorithm::CN_PICO_0>().mask();

        patchCode(cn_trtl_mainloop_ivybridge_asm,            cnv2_mainloop_ivybridge_asm,           ITER,   MASK);
        patchCode(cn_trtl_mainloop_ryzen_asm,                cnv2_mainloop_ryzen_asm,               ITER,   MASK);
        patchCode(cn_trtl_mainloop_bulldozer_asm,            cnv2_mainloop_bulldozer_asm,           ITER,   MASK);
        patchCode(cn_trtl_double_mainloop_sandybridge_asm,   cnv2_double_mainloop_sandybridge_asm,  ITER,   MASK);
    }

    {
        constexpr uint32_t ITER = CnAlgo<Algorithm::CN_PICO_TLO>().iterations();
        constexpr uint32_t MASK = CnAlgo<Algorithm::CN_PICO_TLO>().mask();

        patchCode(cn_tlo_mainloop_ivybridge_asm,             cnv2_mainloop_ivybridge_asm,           ITER,   MASK);
        patchCode(cn_tlo_mainloop_ryzen_asm,                 cnv2_mainloop_ryzen_asm,               ITER,   MASK);
        patchCode(cn_tlo_mainloop_bulldozer_asm,             cnv2_mainloop_bulldozer_asm,           ITER,   MASK);
        patchCode(cn_tlo_double_mainloop_sandybridge_asm,    cnv2_double_mainloop_sandybridge_asm,  ITER,   MASK);
    }
#   endif

    {
        constexpr uint32_t ITER = CnAlgo<Algorithm::CN_ZLS>().iterations();

        patchCode(cn_zls_mainloop_ivybridge_asm,             cnv2_mainloop_ivybridge_asm,           ITER);
        patchCode(cn_zls_mainloop_ryzen_asm,                 cnv2_mainloop_ryzen_asm,               ITER);
        patchCode(cn_zls_mainloop_bulldozer_asm,             cnv2_mainloop_bulldozer_asm,           ITER);
        patchCode(cn_zls_double_mainloop_sandybridge_asm,    cnv2_double_mainloop_sandybridge_asm,  ITER);
    }

    {
        constexpr uint32_t ITER = CnAlgo<Algorithm::CN_DOUBLE>().iterations();

        patchCode(cn_double_mainloop_ivybridge_asm,          cnv2_mainloop_ivybridge_asm,           ITER);
        patchCode(cn_double_mainloop_ryzen_asm,              cnv2_mainloop_ryzen_asm,               ITER);
        patchCode(cn_double_mainloop_bulldozer_asm,          cnv2_mainloop_bulldozer_asm,           ITER);
        patchCode(cn_double_double_mainloop_sandybridge_asm, cnv2_double_mainloop_sandybridge_asm,  ITER);
    }

#   ifdef XMRIG_ALGO_CN_FEMTO
    {
        constexpr uint32_t ITER = CnAlgo<Algorithm::CN_UPX2>().iterations();
        constexpr uint32_t MASK = CnAlgo<Algorithm::CN_UPX2>().mask();

        patchCode<Algorithm::CN_RWZ>(cn_upx2_mainloop_asm,        cnv2_rwz_mainloop_asm,            ITER,   MASK);
        patchCode<Algorithm::CN_RWZ>(cn_upx2_double_mainloop_asm, cnv2_rwz_double_mainloop_asm,     ITER,   MASK);
    }
#endif

    VirtualMemory::protectRX(base, allocation_size);
    VirtualMemory::flushInstructionCache(base, allocation_size);
}
} // namespace xmrig
#else
#   define ADD_FN_ASM(algo)
#endif


static const xmrig::CnHash cnHash;


xmrig::CnHash::CnHash()
{
    ADD_FN(Algorithm::CN_0);
    ADD_FN(Algorithm::CN_1);
    ADD_FN(Algorithm::CN_2);
    ADD_FN(Algorithm::CN_R);
    ADD_FN(Algorithm::CN_FAST);
    ADD_FN(Algorithm::CN_HALF);
    ADD_FN(Algorithm::CN_XAO);
    ADD_FN(Algorithm::CN_RTO);
    ADD_FN(Algorithm::CN_RWZ);
    ADD_FN(Algorithm::CN_ZLS);
    ADD_FN(Algorithm::CN_DOUBLE);

    ADD_FN_ASM(Algorithm::CN_2);
    ADD_FN_ASM(Algorithm::CN_HALF);
    ADD_FN_ASM(Algorithm::CN_R);
    ADD_FN_ASM(Algorithm::CN_RWZ);
    ADD_FN_ASM(Algorithm::CN_ZLS);
    ADD_FN_ASM(Algorithm::CN_DOUBLE);

#   ifdef XMRIG_ALGO_CN_LITE
    ADD_FN(Algorithm::CN_LITE_0);
    ADD_FN(Algorithm::CN_LITE_1);
#   endif

#   ifdef XMRIG_ALGO_CN_HEAVY
    ADD_FN(Algorithm::CN_HEAVY_0);
    ADD_FN(Algorithm::CN_HEAVY_TUBE);
    ADD_FN(Algorithm::CN_HEAVY_XHV);
#   endif

#   ifdef XMRIG_ALGO_CN_PICO
    ADD_FN(Algorithm::CN_PICO_0);
    ADD_FN_ASM(Algorithm::CN_PICO_0);
    ADD_FN(Algorithm::CN_PICO_TLO);
    ADD_FN_ASM(Algorithm::CN_PICO_TLO);
#   endif

    ADD_FN(Algorithm::CN_CCX);

#   ifdef XMRIG_ALGO_CN_FEMTO
    ADD_FN(Algorithm::CN_UPX2);
    ADD_FN_ASM(Algorithm::CN_UPX2);
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    m_map[Algorithm::AR2_CHUKWA] = new cn_hash_fun_array{};
    m_map[Algorithm::AR2_CHUKWA]->data[AV_SINGLE][Assembly::NONE]         = argon2::single_hash<Algorithm::AR2_CHUKWA>;
    m_map[Algorithm::AR2_CHUKWA]->data[AV_SINGLE_SOFT][Assembly::NONE]    = argon2::single_hash<Algorithm::AR2_CHUKWA>;

    m_map[Algorithm::AR2_CHUKWA_V2] = new cn_hash_fun_array{};
    m_map[Algorithm::AR2_CHUKWA_V2]->data[AV_SINGLE][Assembly::NONE]      = argon2::single_hash<Algorithm::AR2_CHUKWA_V2>;
    m_map[Algorithm::AR2_CHUKWA_V2]->data[AV_SINGLE_SOFT][Assembly::NONE] = argon2::single_hash<Algorithm::AR2_CHUKWA_V2>;

    m_map[Algorithm::AR2_WRKZ] = new cn_hash_fun_array{};
    m_map[Algorithm::AR2_WRKZ]->data[AV_SINGLE][Assembly::NONE]           = argon2::single_hash<Algorithm::AR2_WRKZ>;
    m_map[Algorithm::AR2_WRKZ]->data[AV_SINGLE_SOFT][Assembly::NONE]      = argon2::single_hash<Algorithm::AR2_WRKZ>;
#   endif

#   ifdef XMRIG_ALGO_ASTROBWT
    m_map[Algorithm::ASTROBWT_DERO] = new cn_hash_fun_array{};
    m_map[Algorithm::ASTROBWT_DERO]->data[AV_SINGLE][Assembly::NONE]      = astrobwt::single_hash<Algorithm::ASTROBWT_DERO>;
    m_map[Algorithm::ASTROBWT_DERO]->data[AV_SINGLE_SOFT][Assembly::NONE] = astrobwt::single_hash<Algorithm::ASTROBWT_DERO>;
#   endif

#   ifdef XMRIG_ALGO_CN_GPU
    m_map[Algorithm::CN_GPU] = new cn_hash_fun_array{};
    m_map[Algorithm::CN_GPU]->data[AV_SINGLE][Assembly::NONE]      = cryptonight_single_hash_gpu<Algorithm::CN_GPU, false>;
    m_map[Algorithm::CN_GPU]->data[AV_SINGLE_SOFT][Assembly::NONE] = cryptonight_single_hash_gpu<Algorithm::CN_GPU, true>;
#   endif

#   ifdef XMRIG_ALGO_GHOSTRIDER
    ADD_FN(Algorithm::CN_GR_0);
    ADD_FN(Algorithm::CN_GR_1);
    ADD_FN(Algorithm::CN_GR_2);
    ADD_FN(Algorithm::CN_GR_3);
    ADD_FN(Algorithm::CN_GR_4);
    ADD_FN(Algorithm::CN_GR_5);
#   endif

#   ifdef XMRIG_FEATURE_ASM
    patchAsmVariants();
#   endif
}


xmrig::CnHash::~CnHash()
{
    for (auto const& x : m_map) {
      delete m_map[x.first];
    }
}


xmrig::cn_hash_fun xmrig::CnHash::fn(const Algorithm &algorithm, AlgoVariant av, Assembly::Id assembly)
{
    assert(cnHash.m_map.count(algorithm));

    if (!algorithm.isValid()) {
        return nullptr;
    }

    const auto it = cnHash.m_map.find(algorithm);
    if (it == cnHash.m_map.end()) {
        return nullptr;
    }

#   ifdef XMRIG_ALGO_CN_HEAVY
    // cn-heavy optimization for Zen3 CPUs
    if ((av == AV_SINGLE) && (assembly != Assembly::NONE) && (Cpu::info()->arch() == ICpuInfo::ARCH_ZEN3) && (Cpu::info()->model() == 0x21)) {
        switch (algorithm.id()) {
        case Algorithm::CN_HEAVY_0:
            return cryptonight_single_hash<Algorithm::CN_HEAVY_0, false, 3>;

        case Algorithm::CN_HEAVY_TUBE:
            return cryptonight_single_hash<Algorithm::CN_HEAVY_TUBE, false, 3>;

        case Algorithm::CN_HEAVY_XHV:
            return cryptonight_single_hash<Algorithm::CN_HEAVY_XHV, false, 3>;

        default:
            break;
        }
    }
#   endif

#   ifdef XMRIG_FEATURE_ASM
    cn_hash_fun fun = it->second->data[av][Cpu::assembly(assembly)];
    if (fun) {
        return fun;
    }
#   endif

    return it->second->data[av][Assembly::NONE];
}
