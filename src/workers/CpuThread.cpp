/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#include <assert.h>


#include "base/io/log/Log.h"
#include "common/cpu/Cpu.h"
#include "crypto/Asm.h"
#include "Mem.h"
#include "rapidjson/document.h"
#include "workers/CpuThread.h"


#if defined(XMRIG_ARM)
#   include "crypto/CryptoNight_arm.h"
#else
#   include "crypto/CryptoNight_x86.h"
#endif


xmrig::CpuThread::CpuThread(size_t index, Algo algorithm, AlgoVariant av, Multiway multiway, int64_t affinity, int priority, bool softAES, bool prefetch, Assembly assembly) :
    m_algorithm(algorithm),
    m_av(av),
    m_assembly(assembly),
    m_prefetch(prefetch),
    m_softAES(softAES),
    m_priority(priority),
    m_affinity(affinity),
    m_multiway(multiway),
    m_index(index)
{
}


#ifndef XMRIG_NO_ASM
template<typename T, typename U>
static void patchCode(T dst, U src, const uint32_t iterations, const uint32_t mask)
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
        case xmrig::CRYPTONIGHT_ITER:
            *(uint32_t*)(patched_data + i) = iterations;
            break;

        case xmrig::CRYPTONIGHT_MASK:
            *(uint32_t*)(patched_data + i) = mask;
            break;
        }
    }
}


extern "C" void cnv2_mainloop_ivybridge_asm(cryptonight_ctx **ctx);
extern "C" void cnv2_mainloop_ryzen_asm(cryptonight_ctx **ctx);
extern "C" void cnv2_mainloop_bulldozer_asm(cryptonight_ctx **ctx);
extern "C" void cnv2_double_mainloop_sandybridge_asm(cryptonight_ctx **ctx);


xmrig::CpuThread::cn_mainloop_fun        cn_half_mainloop_ivybridge_asm             = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_half_mainloop_ryzen_asm                 = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_half_mainloop_bulldozer_asm             = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_half_double_mainloop_sandybridge_asm    = nullptr;

xmrig::CpuThread::cn_mainloop_fun        cn_trtl_mainloop_ivybridge_asm             = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_trtl_mainloop_ryzen_asm                 = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_trtl_mainloop_bulldozer_asm             = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_trtl_double_mainloop_sandybridge_asm    = nullptr;

xmrig::CpuThread::cn_mainloop_fun        cn_zls_mainloop_ivybridge_asm              = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_zls_mainloop_ryzen_asm                  = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_zls_mainloop_bulldozer_asm              = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_zls_double_mainloop_sandybridge_asm     = nullptr;

xmrig::CpuThread::cn_mainloop_fun        cn_double_mainloop_ivybridge_asm           = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_double_mainloop_ryzen_asm               = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_double_mainloop_bulldozer_asm           = nullptr;
xmrig::CpuThread::cn_mainloop_fun        cn_double_double_mainloop_sandybridge_asm  = nullptr;


void xmrig::CpuThread::patchAsmVariants()
{
    const int allocation_size = 65536;
    uint8_t *base = static_cast<uint8_t *>(Mem::allocateExecutableMemory(allocation_size));

    cn_half_mainloop_ivybridge_asm              = reinterpret_cast<cn_mainloop_fun>         (base + 0x0000);
    cn_half_mainloop_ryzen_asm                  = reinterpret_cast<cn_mainloop_fun>         (base + 0x1000);
    cn_half_mainloop_bulldozer_asm              = reinterpret_cast<cn_mainloop_fun>         (base + 0x2000);
    cn_half_double_mainloop_sandybridge_asm     = reinterpret_cast<cn_mainloop_fun>         (base + 0x3000);

    cn_trtl_mainloop_ivybridge_asm              = reinterpret_cast<cn_mainloop_fun>         (base + 0x4000);
    cn_trtl_mainloop_ryzen_asm                  = reinterpret_cast<cn_mainloop_fun>         (base + 0x5000);
    cn_trtl_mainloop_bulldozer_asm              = reinterpret_cast<cn_mainloop_fun>         (base + 0x6000);
    cn_trtl_double_mainloop_sandybridge_asm     = reinterpret_cast<cn_mainloop_fun>         (base + 0x7000);

    cn_zls_mainloop_ivybridge_asm               = reinterpret_cast<cn_mainloop_fun>         (base + 0x8000);
    cn_zls_mainloop_ryzen_asm                   = reinterpret_cast<cn_mainloop_fun>         (base + 0x9000);
    cn_zls_mainloop_bulldozer_asm               = reinterpret_cast<cn_mainloop_fun>         (base + 0xA000);
    cn_zls_double_mainloop_sandybridge_asm      = reinterpret_cast<cn_mainloop_fun>         (base + 0xB000);

    cn_double_mainloop_ivybridge_asm            = reinterpret_cast<cn_mainloop_fun>         (base + 0xC000);
    cn_double_mainloop_ryzen_asm                = reinterpret_cast<cn_mainloop_fun>         (base + 0xD000);
    cn_double_mainloop_bulldozer_asm            = reinterpret_cast<cn_mainloop_fun>         (base + 0xE000);
    cn_double_double_mainloop_sandybridge_asm   = reinterpret_cast<cn_mainloop_fun>         (base + 0xF000);

    patchCode(cn_half_mainloop_ivybridge_asm,            cnv2_mainloop_ivybridge_asm,           xmrig::CRYPTONIGHT_HALF_ITER,   xmrig::CRYPTONIGHT_MASK);
    patchCode(cn_half_mainloop_ryzen_asm,                cnv2_mainloop_ryzen_asm,               xmrig::CRYPTONIGHT_HALF_ITER,   xmrig::CRYPTONIGHT_MASK);
    patchCode(cn_half_mainloop_bulldozer_asm,            cnv2_mainloop_bulldozer_asm,           xmrig::CRYPTONIGHT_HALF_ITER,   xmrig::CRYPTONIGHT_MASK);
    patchCode(cn_half_double_mainloop_sandybridge_asm,   cnv2_double_mainloop_sandybridge_asm,  xmrig::CRYPTONIGHT_HALF_ITER,   xmrig::CRYPTONIGHT_MASK);

    patchCode(cn_trtl_mainloop_ivybridge_asm,            cnv2_mainloop_ivybridge_asm,           xmrig::CRYPTONIGHT_TRTL_ITER,   xmrig::CRYPTONIGHT_PICO_MASK);
    patchCode(cn_trtl_mainloop_ryzen_asm,                cnv2_mainloop_ryzen_asm,               xmrig::CRYPTONIGHT_TRTL_ITER,   xmrig::CRYPTONIGHT_PICO_MASK);
    patchCode(cn_trtl_mainloop_bulldozer_asm,            cnv2_mainloop_bulldozer_asm,           xmrig::CRYPTONIGHT_TRTL_ITER,   xmrig::CRYPTONIGHT_PICO_MASK);
    patchCode(cn_trtl_double_mainloop_sandybridge_asm,   cnv2_double_mainloop_sandybridge_asm,  xmrig::CRYPTONIGHT_TRTL_ITER,   xmrig::CRYPTONIGHT_PICO_MASK);

    patchCode(cn_zls_mainloop_ivybridge_asm,             cnv2_mainloop_ivybridge_asm,           xmrig::CRYPTONIGHT_ZLS_ITER,    xmrig::CRYPTONIGHT_MASK);
    patchCode(cn_zls_mainloop_ryzen_asm,                 cnv2_mainloop_ryzen_asm,               xmrig::CRYPTONIGHT_ZLS_ITER,    xmrig::CRYPTONIGHT_MASK);
    patchCode(cn_zls_mainloop_bulldozer_asm,             cnv2_mainloop_bulldozer_asm,           xmrig::CRYPTONIGHT_ZLS_ITER,    xmrig::CRYPTONIGHT_MASK);
    patchCode(cn_zls_double_mainloop_sandybridge_asm,    cnv2_double_mainloop_sandybridge_asm,  xmrig::CRYPTONIGHT_ZLS_ITER,    xmrig::CRYPTONIGHT_MASK);

    patchCode(cn_double_mainloop_ivybridge_asm,          cnv2_mainloop_ivybridge_asm,           xmrig::CRYPTONIGHT_DOUBLE_ITER, xmrig::CRYPTONIGHT_MASK);
    patchCode(cn_double_mainloop_ryzen_asm,              cnv2_mainloop_ryzen_asm,               xmrig::CRYPTONIGHT_DOUBLE_ITER, xmrig::CRYPTONIGHT_MASK);
    patchCode(cn_double_mainloop_bulldozer_asm,          cnv2_mainloop_bulldozer_asm,           xmrig::CRYPTONIGHT_DOUBLE_ITER, xmrig::CRYPTONIGHT_MASK);
    patchCode(cn_double_double_mainloop_sandybridge_asm, cnv2_double_mainloop_sandybridge_asm,  xmrig::CRYPTONIGHT_DOUBLE_ITER, xmrig::CRYPTONIGHT_MASK);

    Mem::protectExecutableMemory(base, allocation_size);
    Mem::flushInstructionCache(base, allocation_size);
}
#endif


bool xmrig::CpuThread::isSoftAES(AlgoVariant av)
{
    return av == AV_SINGLE_SOFT || av == AV_DOUBLE_SOFT || av > AV_PENTA;
}


#ifndef XMRIG_NO_ASM
template<xmrig::Algo algo, xmrig::Variant variant>
static inline void add_asm_func(xmrig::CpuThread::cn_hash_fun(&asm_func_map)[xmrig::ALGO_MAX][xmrig::AV_MAX][xmrig::VARIANT_MAX][xmrig::ASM_MAX])
{
    asm_func_map[algo][xmrig::AV_SINGLE][variant][xmrig::ASM_INTEL]     = cryptonight_single_hash_asm<algo, variant, xmrig::ASM_INTEL>;
    asm_func_map[algo][xmrig::AV_SINGLE][variant][xmrig::ASM_RYZEN]     = cryptonight_single_hash_asm<algo, variant, xmrig::ASM_RYZEN>;
    asm_func_map[algo][xmrig::AV_SINGLE][variant][xmrig::ASM_BULLDOZER] = cryptonight_single_hash_asm<algo, variant, xmrig::ASM_BULLDOZER>;

    asm_func_map[algo][xmrig::AV_DOUBLE][variant][xmrig::ASM_INTEL]     = cryptonight_double_hash_asm<algo, variant, xmrig::ASM_INTEL>;
    asm_func_map[algo][xmrig::AV_DOUBLE][variant][xmrig::ASM_RYZEN]     = cryptonight_double_hash_asm<algo, variant, xmrig::ASM_RYZEN>;
    asm_func_map[algo][xmrig::AV_DOUBLE][variant][xmrig::ASM_BULLDOZER] = cryptonight_double_hash_asm<algo, variant, xmrig::ASM_BULLDOZER>;
}
#endif

xmrig::CpuThread::cn_hash_fun xmrig::CpuThread::fn(Algo algorithm, AlgoVariant av, Variant variant, Assembly assembly)
{
    assert(variant >= VARIANT_0 && variant < VARIANT_MAX);

#   ifndef XMRIG_NO_ASM
    if (assembly == ASM_AUTO) {
        assembly = Cpu::info()->assembly();
    }

    static cn_hash_fun asm_func_map[ALGO_MAX][AV_MAX][VARIANT_MAX][ASM_MAX] = {};
    static bool asm_func_map_initialized = false;

    if (!asm_func_map_initialized) {
        add_asm_func<CRYPTONIGHT, VARIANT_2>(asm_func_map);
        add_asm_func<CRYPTONIGHT, VARIANT_HALF>(asm_func_map);
        add_asm_func<CRYPTONIGHT, VARIANT_WOW>(asm_func_map);
        add_asm_func<CRYPTONIGHT, VARIANT_4>(asm_func_map);

#       ifndef XMRIG_NO_CN_PICO
        add_asm_func<CRYPTONIGHT_PICO, VARIANT_TRTL>(asm_func_map);
#       endif

        add_asm_func<CRYPTONIGHT, VARIANT_RWZ>(asm_func_map);
        add_asm_func<CRYPTONIGHT, VARIANT_ZLS>(asm_func_map);
        add_asm_func<CRYPTONIGHT, VARIANT_DOUBLE>(asm_func_map);

        asm_func_map_initialized = true;
    }

    cn_hash_fun fun = asm_func_map[algorithm][av][variant][assembly];
    if (fun) {
        return fun;
    }
#   endif

    constexpr const size_t count = VARIANT_MAX * 10 * ALGO_MAX;

    static const cn_hash_fun func_table[] = {
        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_0>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_0>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_0>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_0>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_0>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_0>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_0>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_0>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_0>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_0>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_1>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_1>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_1>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_1>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_1>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_1>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_1>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_1>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_1>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_1>,

        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TUBE

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_XTL>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_XTL>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_XTL>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_XTL>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_XTL>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_XTL>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_XTL>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_XTL>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_XTL>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_XTL>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_MSR>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_MSR>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_MSR>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_MSR>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_MSR>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_MSR>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_MSR>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_MSR>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_MSR>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_MSR>,

        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XHV

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_XAO>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_XAO>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_XAO>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_XAO>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_XAO>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_XAO>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_XAO>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_XAO>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_XAO>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_XAO>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_RTO>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_RTO>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_RTO>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_RTO>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_RTO>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_RTO>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_RTO>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_RTO>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_RTO>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_RTO>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_2>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_2>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_2>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_2>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_2>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_2>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_2>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_2>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_2>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_2>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_HALF>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_HALF>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_HALF>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_HALF>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_HALF>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_HALF>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_HALF>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_HALF>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_HALF>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_HALF>,

        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TRTL

#       ifndef XMRIG_NO_CN_GPU
        cryptonight_single_hash_gpu<CRYPTONIGHT, false, VARIANT_GPU>,
        nullptr,
        cryptonight_single_hash_gpu<CRYPTONIGHT, true,  VARIANT_GPU>,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
#       else
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_GPU
#       endif

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_WOW>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_WOW>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_WOW>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_WOW>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_WOW>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_WOW>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_WOW>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_WOW>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_WOW>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_WOW>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_4>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_4>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_4>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_4>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_4>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_4>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_4>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_4>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_4>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_4>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_RWZ>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_RWZ>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_RWZ>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_RWZ>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_RWZ>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_RWZ>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_RWZ>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_RWZ>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_RWZ>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_RWZ>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_ZLS>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_ZLS>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_ZLS>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_ZLS>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_ZLS>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_ZLS>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_ZLS>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_ZLS>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_ZLS>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_ZLS>,

        cryptonight_single_hash<CRYPTONIGHT, false, VARIANT_DOUBLE>,
        cryptonight_double_hash<CRYPTONIGHT, false, VARIANT_DOUBLE>,
        cryptonight_single_hash<CRYPTONIGHT, true,  VARIANT_DOUBLE>,
        cryptonight_double_hash<CRYPTONIGHT, true,  VARIANT_DOUBLE>,
        cryptonight_triple_hash<CRYPTONIGHT, false, VARIANT_DOUBLE>,
        cryptonight_quad_hash<CRYPTONIGHT,   false, VARIANT_DOUBLE>,
        cryptonight_penta_hash<CRYPTONIGHT,  false, VARIANT_DOUBLE>,
        cryptonight_triple_hash<CRYPTONIGHT, true,  VARIANT_DOUBLE>,
        cryptonight_quad_hash<CRYPTONIGHT,   true,  VARIANT_DOUBLE>,
        cryptonight_penta_hash<CRYPTONIGHT,  true,  VARIANT_DOUBLE>,

#       ifndef XMRIG_NO_AEON
        cryptonight_single_hash<CRYPTONIGHT_LITE, false, VARIANT_0>,
        cryptonight_double_hash<CRYPTONIGHT_LITE, false, VARIANT_0>,
        cryptonight_single_hash<CRYPTONIGHT_LITE, true,  VARIANT_0>,
        cryptonight_double_hash<CRYPTONIGHT_LITE, true,  VARIANT_0>,
        cryptonight_triple_hash<CRYPTONIGHT_LITE, false, VARIANT_0>,
        cryptonight_quad_hash<CRYPTONIGHT_LITE,   false, VARIANT_0>,
        cryptonight_penta_hash<CRYPTONIGHT_LITE,  false, VARIANT_0>,
        cryptonight_triple_hash<CRYPTONIGHT_LITE, true,  VARIANT_0>,
        cryptonight_quad_hash<CRYPTONIGHT_LITE,   true,  VARIANT_0>,
        cryptonight_penta_hash<CRYPTONIGHT_LITE,  true,  VARIANT_0>,

        cryptonight_single_hash<CRYPTONIGHT_LITE, false, VARIANT_1>,
        cryptonight_double_hash<CRYPTONIGHT_LITE, false, VARIANT_1>,
        cryptonight_single_hash<CRYPTONIGHT_LITE, true,  VARIANT_1>,
        cryptonight_double_hash<CRYPTONIGHT_LITE, true,  VARIANT_1>,
        cryptonight_triple_hash<CRYPTONIGHT_LITE, false, VARIANT_1>,
        cryptonight_quad_hash<CRYPTONIGHT_LITE,   false, VARIANT_1>,
        cryptonight_penta_hash<CRYPTONIGHT_LITE,  false, VARIANT_1>,
        cryptonight_triple_hash<CRYPTONIGHT_LITE, true,  VARIANT_1>,
        cryptonight_quad_hash<CRYPTONIGHT_LITE,   true,  VARIANT_1>,
        cryptonight_penta_hash<CRYPTONIGHT_LITE,  true,  VARIANT_1>,

        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TUBE
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_MSR
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XHV
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XAO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RTO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_2
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_HALF
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TRTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_GPU
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_WOW
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_4
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RWZ
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_ZLS
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_DOUBLE
#       else
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_0
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_1
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TUBE
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_MSR
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XHV
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XAO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RTO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_2
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_HALF
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TRTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_GPU
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_WOW
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_4
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RWZ
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_ZLS
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_DOUBLE
#       endif

#       ifndef XMRIG_NO_SUMO
        cryptonight_single_hash<CRYPTONIGHT_HEAVY, false, VARIANT_0>,
        cryptonight_double_hash<CRYPTONIGHT_HEAVY, false, VARIANT_0>,
        cryptonight_single_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_0>,
        cryptonight_double_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_0>,
        cryptonight_triple_hash<CRYPTONIGHT_HEAVY, false, VARIANT_0>,
        cryptonight_quad_hash<CRYPTONIGHT_HEAVY,   false, VARIANT_0>,
        cryptonight_penta_hash<CRYPTONIGHT_HEAVY,  false, VARIANT_0>,
        cryptonight_triple_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_0>,
        cryptonight_quad_hash<CRYPTONIGHT_HEAVY,   true,  VARIANT_0>,
        cryptonight_penta_hash<CRYPTONIGHT_HEAVY,  true,  VARIANT_0>,

        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_1

        cryptonight_single_hash<CRYPTONIGHT_HEAVY, false, VARIANT_TUBE>,
        cryptonight_double_hash<CRYPTONIGHT_HEAVY, false, VARIANT_TUBE>,
        cryptonight_single_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_TUBE>,
        cryptonight_double_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_TUBE>,
        cryptonight_triple_hash<CRYPTONIGHT_HEAVY, false, VARIANT_TUBE>,
        cryptonight_quad_hash<CRYPTONIGHT_HEAVY,   false, VARIANT_TUBE>,
        cryptonight_penta_hash<CRYPTONIGHT_HEAVY,  false, VARIANT_TUBE>,
        cryptonight_triple_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_TUBE>,
        cryptonight_quad_hash<CRYPTONIGHT_HEAVY,   true,  VARIANT_TUBE>,
        cryptonight_penta_hash<CRYPTONIGHT_HEAVY,  true,  VARIANT_TUBE>,

        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_MSR

        cryptonight_single_hash<CRYPTONIGHT_HEAVY, false, VARIANT_XHV>,
        cryptonight_double_hash<CRYPTONIGHT_HEAVY, false, VARIANT_XHV>,
        cryptonight_single_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_XHV>,
        cryptonight_double_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_XHV>,
        cryptonight_triple_hash<CRYPTONIGHT_HEAVY, false, VARIANT_XHV>,
        cryptonight_quad_hash<CRYPTONIGHT_HEAVY,   false, VARIANT_XHV>,
        cryptonight_penta_hash<CRYPTONIGHT_HEAVY,  false, VARIANT_XHV>,
        cryptonight_triple_hash<CRYPTONIGHT_HEAVY, true,  VARIANT_XHV>,
        cryptonight_quad_hash<CRYPTONIGHT_HEAVY,   true,  VARIANT_XHV>,
        cryptonight_penta_hash<CRYPTONIGHT_HEAVY,  true,  VARIANT_XHV>,

        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XAO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RTO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_2
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_HALF
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TRTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_GPU
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_WOW
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_4
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RWZ
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_ZLS
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_DOUBLE
#       else
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_0
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_1
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TUBE
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_MSR
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XHV
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XAO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RTO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_2
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_HALF
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TRTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_GPU
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_WOW
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_4
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RWZ
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_ZLS
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_DOUBLE
#       endif

#       ifndef XMRIG_NO_CN_PICO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_0
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_1
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TUBE
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_MSR
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XHV
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XAO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RTO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_2
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_HALF

        cryptonight_single_hash<CRYPTONIGHT_PICO, false, VARIANT_TRTL>,
        cryptonight_double_hash<CRYPTONIGHT_PICO, false, VARIANT_TRTL>,
        cryptonight_single_hash<CRYPTONIGHT_PICO, true,  VARIANT_TRTL>,
        cryptonight_double_hash<CRYPTONIGHT_PICO, true,  VARIANT_TRTL>,
        cryptonight_triple_hash<CRYPTONIGHT_PICO, false, VARIANT_TRTL>,
        cryptonight_quad_hash<CRYPTONIGHT_PICO,   false, VARIANT_TRTL>,
        cryptonight_penta_hash<CRYPTONIGHT_PICO,  false, VARIANT_TRTL>,
        cryptonight_triple_hash<CRYPTONIGHT_PICO, true,  VARIANT_TRTL>,
        cryptonight_quad_hash<CRYPTONIGHT_PICO,   true,  VARIANT_TRTL>,
        cryptonight_penta_hash<CRYPTONIGHT_PICO,  true,  VARIANT_TRTL>,

        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_GPU
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_WOW
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_4
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RWZ
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_ZLS
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_DOUBLE
#       else
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_0
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_1
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TUBE
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_MSR
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XHV
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_XAO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RTO
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_2
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_HALF
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_TRTL
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_GPU
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_WOW
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_4
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_RWZ
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_ZLS
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // VARIANT_DOUBLE
#       endif
    };

    static_assert(count == sizeof(func_table) / sizeof(func_table[0]), "func_table size mismatch");

    const size_t index = VARIANT_MAX * 10 * algorithm + 10 * variant + av - 1;

#   ifndef NDEBUG
    cn_hash_fun func = func_table[index];

    assert(index < sizeof(func_table) / sizeof(func_table[0]));
    assert(func != nullptr);

    return func;
#   else
    return func_table[index];
#   endif
}


xmrig::CpuThread *xmrig::CpuThread::createFromAV(size_t index, Algo algorithm, AlgoVariant av, int64_t affinity, int priority, Assembly assembly)
{
    assert(av > AV_AUTO && av < AV_MAX);

    int64_t cpuId = -1L;

    if (affinity != -1L) {
        size_t idx = 0;

        for (size_t i = 0; i < 64; i++) {
            if (!(affinity & (1ULL << i))) {
                continue;
            }

            if (idx == index) {
                cpuId = i;
                break;
            }

            idx++;
        }
    }

    return new CpuThread(index, algorithm, av, multiway(av), cpuId, priority, isSoftAES(av), false, assembly);
}


xmrig::CpuThread *xmrig::CpuThread::createFromData(size_t index, Algo algorithm, const CpuThread::Data &data, int priority, bool softAES)
{
    int av                  = AV_AUTO;
    const Multiway multiway = data.multiway;

    if (multiway <= DoubleWay) {
        av = softAES ? (multiway + 2) : multiway;
    }
    else {
        av = softAES ? (multiway + 5) : (multiway + 2);
    }

    assert(av > AV_AUTO && av < AV_MAX);

    return new CpuThread(index, algorithm, static_cast<AlgoVariant>(av), multiway, data.affinity, priority, softAES, false, data.assembly);
}


xmrig::CpuThread::Data xmrig::CpuThread::parse(const rapidjson::Value &object)
{
    Data data;

    const auto &multiway = object["low_power_mode"];
    if (multiway.IsBool()) {
        data.multiway = multiway.IsTrue() ? DoubleWay : SingleWay;
        data.valid    = true;
    }
    else if (multiway.IsUint()) {
        data.setMultiway(multiway.GetInt());
    }

    if (!data.valid) {
        return data;
    }

    const auto &affinity = object["affine_to_cpu"];
    if (affinity.IsUint64()) {
        data.affinity = affinity.GetInt64();
    }

#   ifndef XMRIG_NO_ASM
    data.assembly = Asm::parse(object["asm"]);
#   endif

    return data;
}


xmrig::IThread::Multiway xmrig::CpuThread::multiway(AlgoVariant av)
{
    switch (av) {
    case AV_SINGLE:
    case AV_SINGLE_SOFT:
        return SingleWay;

    case AV_DOUBLE_SOFT:
    case AV_DOUBLE:
        return DoubleWay;

    case AV_TRIPLE_SOFT:
    case AV_TRIPLE:
        return TripleWay;

    case AV_QUAD_SOFT:
    case AV_QUAD:
        return QuadWay;

    case AV_PENTA_SOFT:
    case AV_PENTA:
        return PentaWay;

    default:
        break;
    }

    return SingleWay;
}


#ifdef APP_DEBUG
void xmrig::CpuThread::print() const
{
    LOG_DEBUG(GREEN_BOLD("CPU thread:   ") " index " WHITE_BOLD("%zu") ", multiway " WHITE_BOLD("%d") ", av " WHITE_BOLD("%d") ",",
              index(), static_cast<int>(multiway()), static_cast<int>(m_av));

#   ifndef XMRIG_NO_ASM
    LOG_DEBUG("               assembly: %s, affine_to_cpu: %" PRId64, Asm::toString(m_assembly), affinity());
#   else
    LOG_DEBUG("               affine_to_cpu: %" PRId64, affinity());
#   endif
}
#endif


#ifdef XMRIG_FEATURE_API
rapidjson::Value xmrig::CpuThread::toAPI(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    Value obj(kObjectType);
    auto &allocator = doc.GetAllocator();

    obj.AddMember("type",          "cpu", allocator);
    obj.AddMember("av",             m_av, allocator);
    obj.AddMember("low_power_mode", multiway(), allocator);
    obj.AddMember("affine_to_cpu",  affinity(), allocator);
    obj.AddMember("priority",       priority(), allocator);
    obj.AddMember("soft_aes",       isSoftAES(), allocator);

    return obj;
}
#endif


rapidjson::Value xmrig::CpuThread::toConfig(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    Value obj(kObjectType);
    auto &allocator = doc.GetAllocator();

    obj.AddMember("low_power_mode", multiway(), allocator);
    obj.AddMember("affine_to_cpu",  affinity() == -1L ? Value(kFalseType) : Value(affinity()), allocator);

#   ifndef XMRIG_NO_ASM
    obj.AddMember("asm", Asm::toJSON(m_assembly), allocator);
#   endif

    return obj;
}
