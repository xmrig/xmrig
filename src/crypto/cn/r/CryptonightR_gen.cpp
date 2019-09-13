/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#include <cstring>
#include "crypto/cn/CryptoNight_monero.h"

typedef void(*void_func)();

#include "crypto/cn/asm/CryptonightR_template.h"
#include "crypto/common/Assembly.h"
#include "crypto/common/VirtualMemory.h"


static inline void add_code(uint8_t* &p, void (*p1)(), void (*p2)())
{
    const ptrdiff_t size = reinterpret_cast<const uint8_t*>(p2) - reinterpret_cast<const uint8_t*>(p1);
    if (size > 0) {
        memcpy(p, reinterpret_cast<void*>(p1), size);
        p += size;
    }
}

static inline void add_random_math(uint8_t* &p, const V4_Instruction* code, int code_size, const void_func* instructions, const void_func* instructions_mov, bool is_64_bit, xmrig::Assembly::Id ASM)
{
    uint32_t prev_rot_src = (uint32_t)(-1);

    for (int i = 0;; ++i) {
        const V4_Instruction inst = code[i];
        if (inst.opcode == RET) {
            break;
        }

        uint8_t opcode = (inst.opcode == MUL) ? inst.opcode : (inst.opcode + 2);
        uint8_t dst_index = inst.dst_index;
        uint8_t src_index = inst.src_index;

        const uint32_t a = inst.dst_index;
        const uint32_t b = inst.src_index;
        const uint8_t c = opcode | (dst_index << V4_OPCODE_BITS) | (((src_index == 8) ? dst_index : src_index) << (V4_OPCODE_BITS + V4_DST_INDEX_BITS));

        switch (inst.opcode) {
        case ROR:
        case ROL:
            if (b != prev_rot_src) {
                prev_rot_src = b;
                add_code(p, instructions_mov[c], instructions_mov[c + 1]);
            }
            break;
        }

        if (a == prev_rot_src) {
            prev_rot_src = (uint32_t)(-1);
        }

        void_func begin = instructions[c];

        if ((ASM = xmrig::Assembly::BULLDOZER) && (inst.opcode == MUL) && !is_64_bit) {
            // AMD Bulldozer has latency 4 for 32-bit IMUL and 6 for 64-bit IMUL
            // Always use 32-bit IMUL for AMD Bulldozer in 32-bit mode - skip prefix 0x48 and change 0x49 to 0x41
            uint8_t* prefix = reinterpret_cast<uint8_t*>(begin);

            if (*prefix == 0x49) {
                *(p++) = 0x41;
            }

            begin = reinterpret_cast<void_func>(prefix + 1);
        }

        add_code(p, begin, instructions[c + 1]);

        if (inst.opcode == ADD) {
            *(uint32_t*)(p - sizeof(uint32_t) - (is_64_bit ? 3 : 0)) = inst.C;
            if (is_64_bit) {
                prev_rot_src = (uint32_t)(-1);
            }
        }
    }
}

void v4_compile_code(const V4_Instruction* code, int code_size, void* machine_code, xmrig::Assembly ASM)
{
    uint8_t* p0 = reinterpret_cast<uint8_t*>(machine_code);
    uint8_t* p = p0;

    add_code(p, CryptonightR_template_part1, CryptonightR_template_part2);
    add_random_math(p, code, code_size, instructions, instructions_mov, false, ASM);
    add_code(p, CryptonightR_template_part2, CryptonightR_template_part3);
    *(int*)(p - 4) = static_cast<int>((((const uint8_t*)CryptonightR_template_mainloop) - ((const uint8_t*)CryptonightR_template_part1)) - (p - p0));
    add_code(p, CryptonightR_template_part3, CryptonightR_template_end);

    xmrig::VirtualMemory::flushInstructionCache(machine_code, p - p0);
}

void v4_compile_code_double(const V4_Instruction* code, int code_size, void* machine_code, xmrig::Assembly ASM)
{
    uint8_t* p0 = reinterpret_cast<uint8_t*>(machine_code);
    uint8_t* p = p0;

    add_code(p, CryptonightR_template_double_part1, CryptonightR_template_double_part2);
    add_random_math(p, code, code_size, instructions, instructions_mov, false, ASM);
    add_code(p, CryptonightR_template_double_part2, CryptonightR_template_double_part3);
    add_random_math(p, code, code_size, instructions, instructions_mov, false, ASM);
    add_code(p, CryptonightR_template_double_part3, CryptonightR_template_double_part4);
    *(int*)(p - 4) = static_cast<int>((((const uint8_t*)CryptonightR_template_double_mainloop) - ((const uint8_t*)CryptonightR_template_double_part1)) - (p - p0));
    add_code(p, CryptonightR_template_double_part4, CryptonightR_template_double_end);

    xmrig::VirtualMemory::flushInstructionCache(machine_code, p - p0);
}

void v4_soft_aes_compile_code(const V4_Instruction* code, int code_size, void* machine_code, xmrig::Assembly ASM)
{
    uint8_t* p0 = reinterpret_cast<uint8_t*>(machine_code);
    uint8_t* p = p0;

    add_code(p, CryptonightR_soft_aes_template_part1, CryptonightR_soft_aes_template_part2);
    add_random_math(p, code, code_size, instructions, instructions_mov, false, ASM);
    add_code(p, CryptonightR_soft_aes_template_part2, CryptonightR_soft_aes_template_part3);
    *(int*)(p - 4) = static_cast<int>((((const uint8_t*)CryptonightR_soft_aes_template_mainloop) - ((const uint8_t*)CryptonightR_soft_aes_template_part1)) - (p - p0));
    add_code(p, CryptonightR_soft_aes_template_part3, CryptonightR_soft_aes_template_end);

    xmrig::VirtualMemory::flushInstructionCache(machine_code, p - p0);
}
