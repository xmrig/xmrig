/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include <stdio.h>


#include "backend/cpu/Cpu.h"
#include "crypto/cn/CnHash.h"
#include "crypto/common/VirtualMemory.h"


#if defined(XMRIG_ARM)
#   include "crypto/cn/CryptoNight_arm.h"
#else
#   include "crypto/cn/CryptoNight_x86.h"
#endif


#define ADD_FN(algo) \
    m_map[algo][AV_SINGLE][Assembly::NONE]      = cryptonight_single_hash<algo, false>; \
    m_map[algo][AV_SINGLE_SOFT][Assembly::NONE] = cryptonight_single_hash<algo, true>;  \
    m_map[algo][AV_DOUBLE][Assembly::NONE]      = cryptonight_double_hash<algo, false>; \
    m_map[algo][AV_DOUBLE_SOFT][Assembly::NONE] = cryptonight_double_hash<algo, true>;  \
    m_map[algo][AV_TRIPLE][Assembly::NONE]      = cryptonight_triple_hash<algo, false>; \
    m_map[algo][AV_TRIPLE_SOFT][Assembly::NONE] = cryptonight_triple_hash<algo, true>;  \
    m_map[algo][AV_QUAD][Assembly::NONE]        = cryptonight_quad_hash<algo,   false>; \
    m_map[algo][AV_QUAD_SOFT][Assembly::NONE]   = cryptonight_quad_hash<algo,   true>;  \
    m_map[algo][AV_PENTA][Assembly::NONE]       = cryptonight_penta_hash<algo,  false>; \
    m_map[algo][AV_PENTA_SOFT][Assembly::NONE]  = cryptonight_penta_hash<algo,  true>;


#ifdef XMRIG_FEATURE_ASM
#   define ADD_FN_ASM(algo) \
    m_map[algo][AV_SINGLE][Assembly::INTEL]     = cryptonight_single_hash_asm<algo, Assembly::INTEL>;     \
    m_map[algo][AV_SINGLE][Assembly::RYZEN]     = cryptonight_single_hash_asm<algo, Assembly::RYZEN>;     \
    m_map[algo][AV_SINGLE][Assembly::BULLDOZER] = cryptonight_single_hash_asm<algo, Assembly::BULLDOZER>; \
    m_map[algo][AV_DOUBLE][Assembly::INTEL]     = cryptonight_double_hash_asm<algo, Assembly::INTEL>;     \
    m_map[algo][AV_DOUBLE][Assembly::RYZEN]     = cryptonight_double_hash_asm<algo, Assembly::RYZEN>;     \
    m_map[algo][AV_DOUBLE][Assembly::BULLDOZER] = cryptonight_double_hash_asm<algo, Assembly::BULLDOZER>;


extern "C" void cnv2_mainloop_ivybridge_asm(cryptonight_ctx **ctx);
extern "C" void cnv2_mainloop_ryzen_asm(cryptonight_ctx **ctx);
extern "C" void cnv2_mainloop_bulldozer_asm(cryptonight_ctx **ctx);
extern "C" void cnv2_double_mainloop_sandybridge_asm(cryptonight_ctx **ctx);


namespace xmrig {


cn_mainloop_fun        cn_half_mainloop_ivybridge_asm             = nullptr;
cn_mainloop_fun        cn_half_mainloop_ryzen_asm                 = nullptr;
cn_mainloop_fun        cn_half_mainloop_bulldozer_asm             = nullptr;
cn_mainloop_fun        cn_half_double_mainloop_sandybridge_asm    = nullptr;

cn_mainloop_fun        cn_trtl_mainloop_ivybridge_asm             = nullptr;
cn_mainloop_fun        cn_trtl_mainloop_ryzen_asm                 = nullptr;
cn_mainloop_fun        cn_trtl_mainloop_bulldozer_asm             = nullptr;
cn_mainloop_fun        cn_trtl_double_mainloop_sandybridge_asm    = nullptr;

cn_mainloop_fun        cn_zls_mainloop_ivybridge_asm              = nullptr;
cn_mainloop_fun        cn_zls_mainloop_ryzen_asm                  = nullptr;
cn_mainloop_fun        cn_zls_mainloop_bulldozer_asm              = nullptr;
cn_mainloop_fun        cn_zls_double_mainloop_sandybridge_asm     = nullptr;

cn_mainloop_fun        cn_double_mainloop_ivybridge_asm           = nullptr;
cn_mainloop_fun        cn_double_mainloop_ryzen_asm               = nullptr;
cn_mainloop_fun        cn_double_mainloop_bulldozer_asm           = nullptr;
cn_mainloop_fun        cn_double_double_mainloop_sandybridge_asm  = nullptr;


template<typename T, typename U>
static void patchCode(T dst, U src, const uint32_t iterations, const uint32_t mask = CnAlgo<Algorithm::CN_HALF>().mask())
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);

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

    uint8_t* patched_data = reinterpret_cast<uint8_t*>(dst);
    for (size_t i = 0; i + sizeof(uint32_t) <= size; ++i) {
        switch (*(uint32_t*)(patched_data + i)) {
        case CnAlgo<Algorithm::CN_2>().iterations():
            *(uint32_t*)(patched_data + i) = iterations;
            break;

        case CnAlgo<Algorithm::CN_2>().mask():
            *(uint32_t*)(patched_data + i) = mask;
            break;
        }
    }
}


static void patchAsmVariants()
{
    const int allocation_size = 65536;
    uint8_t *base = static_cast<uint8_t *>(VirtualMemory::allocateExecutableMemory(allocation_size));

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

    VirtualMemory::protectExecutableMemory(base, allocation_size);
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
    ADD_FN(Algorithm::CN_WOW);
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
    ADD_FN_ASM(Algorithm::CN_WOW);
    ADD_FN_ASM(Algorithm::CN_RWZ);
    ADD_FN_ASM(Algorithm::CN_ZLS);
    ADD_FN_ASM(Algorithm::CN_DOUBLE);

#   ifdef XMRIG_ALGO_CN_GPU
    m_map[Algorithm::CN_GPU][AV_SINGLE][Assembly::NONE]      = cryptonight_single_hash_gpu<Algorithm::CN_GPU, false>;
    m_map[Algorithm::CN_GPU][AV_SINGLE_SOFT][Assembly::NONE] = cryptonight_single_hash_gpu<Algorithm::CN_GPU, true>;
#   endif

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
#   endif

#   ifdef XMRIG_FEATURE_ASM
    patchAsmVariants();
#   endif
}


xmrig::cn_hash_fun xmrig::CnHash::fn(const Algorithm &algorithm, AlgoVariant av, Assembly::Id assembly)
{
    if (!algorithm.isValid()) {
        return nullptr;
    }

#   ifdef XMRIG_FEATURE_ASM
    cn_hash_fun fun = cnHash.m_map[algorithm][av][Cpu::assembly(assembly)];
    if (fun) {
        return fun;
    }
#   endif

    return cnHash.m_map[algorithm][av][Assembly::NONE];
}
