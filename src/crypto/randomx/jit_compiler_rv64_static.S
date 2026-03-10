/*
Copyright (c) 2023 tevador <tevador@gmail.com>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define DECL(x) x

.text
.option rvc

#include "configuration.h"

.global DECL(randomx_riscv64_literals)
.global DECL(randomx_riscv64_literals_end)
.global DECL(randomx_riscv64_data_init)
.global DECL(randomx_riscv64_fix_data_call)
.global DECL(randomx_riscv64_prologue)
.global DECL(randomx_riscv64_loop_begin)
.global DECL(randomx_riscv64_data_read)
.global DECL(randomx_riscv64_data_read_light)
.global DECL(randomx_riscv64_fix_loop_call)
.global DECL(randomx_riscv64_spad_store)
.global DECL(randomx_riscv64_spad_store_hardaes)
.global DECL(randomx_riscv64_spad_store_softaes)
.global DECL(randomx_riscv64_loop_end)
.global DECL(randomx_riscv64_fix_continue_loop)
.global DECL(randomx_riscv64_epilogue)
.global DECL(randomx_riscv64_softaes)
.global DECL(randomx_riscv64_program_end)
.global DECL(randomx_riscv64_ssh_init)
.global DECL(randomx_riscv64_ssh_load)
.global DECL(randomx_riscv64_ssh_prefetch)
.global DECL(randomx_riscv64_ssh_end)

/* The literal pool can fit at most 494 IMUL_RCP literals */
#if RANDOMX_PROGRAM_SIZE > 494
    #error RANDOMX_PROGRAM_SIZE larger than 494 is not supported.
#endif

#define RANDOMX_ARGON_MEMORY 262144
#define RANDOMX_CACHE_MASK (RANDOMX_ARGON_MEMORY*16-1)

/* shared literal pool: 4 KB */
    /* space for 256 IMUL_RCP literals -2048 */
    /* filled by JIT compiler */
DECL(randomx_riscv64_literals):
literal_pool:
    /* SuperscalarHash constants +0 */
    .dword 6364136223846793005
    .dword 9298411001130361340
    .dword 12065312585734608966
    .dword 9306329213124626780
    .dword 5281919268842080866
    .dword 10536153434571861004
    .dword 3398623926847679864
    .dword 9549104520008361294
    /* CFROUND lookup table +64 */
    .word  0x00000000 /* RTN */
    .word  0x00000002 /* RDN */
    .word  0x00000003 /* RUP */
    .word  0x00000001 /* RTZ */
    /* mask literals +80,+84,+88,+92,+96,+104 */
    .word (/*RANDOMX_SCRATCHPAD_L1*/0) /* filled by JIT compiler */
    .word (/*RANDOMX_SCRATCHPAD_L2*/0) /* filled by JIT compiler */
    .word (/*RANDOMX_SCRATCHPAD_L3*/0) /* filled by JIT compiler */
    .word (/*RANDOMX_DATASET_BASE_SIZE*/0) /* filled by JIT compiler */
    .dword 0x80f0000000000000
    .dword 0x00ffffffffffffff
DECL(randomx_riscv64_literals_end):
    /* E reg. set masks, +112,+120 */
    .dword 0 /* filled by JIT compiler */
    .dword 0 /* filled by JIT compiler */
    /* soft AES table addresses, +128,+136 */
    .dword 0 /* filled by JIT compiler */
    .dword 0 /* filled by JIT compiler */
    /* space for 238 IMUL_RCP literals, +144 */
    .fill 238,8,0 /* filled by JIT compiler */

/* ================================= */
/* Dataset init function entry point */
/* ================================= */

/* Register allocation:
   ----------------------
  x0      -> zero
  x1      -> temp/return address
  x2      -> stack pointer (sp)
  x3      -> literal pool pointer
  x5      -> dataset pointer
  x6      -> cache pointer
  x7      -> temp/itemNumber
  x8-x15  -> SuperscalarHash registers
  x16     -> itemNumber
  x17     -> endItem
  x28-x31 -> temp

  Stack layout:
  ------------------------
  sp+
  0   -> return address
  8   -> saved x3
  16  -> saved x8-x9
  32  -> caller stack
*/
DECL(randomx_riscv64_data_init):
    addi sp, sp, -32
    /* dataset ptr */
    mv x5, x11
    /* cache->memory */
    ld x6, 0(x10)
    /* callee saved registers */
    sd x1, 0(sp)
    sd x3, 8(sp)
    /* literal pool */
    lla x3, literal_pool
    sd x8, 16(sp)
    sd x9, 24(sp)
    /* startItem */
    mv x16, x12
    /* endItem */
    mv x17, x13
init_item:
    mv x7, x16
DECL(randomx_riscv64_fix_data_call):
    jal superscalar_hash /* JIT compiler will adjust the offset */
    sd x8, 0(x5)
    sd x9, 8(x5)
    sd x10, 16(x5)
    sd x11, 24(x5)
    sd x12, 32(x5)
    sd x13, 40(x5)
    sd x14, 48(x5)
    sd x15, 56(x5)
    addi x5, x5, 64
    addi x16, x16, 1
    bltu x16, x17, init_item
    ld x1, 0(sp)
    ld x3, 8(sp)
    ld x8, 16(sp)
    ld x9, 24(sp)
    addi sp, sp, 32
    ret

/* ====================================== */
/* Program execution function entry point */
/* ====================================== */

/* Register allocation:
   ----------------------
  x0      -> zero
  x1      -> temp/scratchpad L3 mask
  x2      -> stack pointer (sp)
  x3      -> literal pool pointer
  x5      -> scratchpad pointer
  x6      -> dataset/cache pointer
  x7      -> temp/next dataset access
  x8      -> temp
  x9      -> temp
  x10     -> scratchpad L1 mask (0x0000000000003ff8)
  x11     -> scratchpad L2 mask (0x000000000003fff8)
  x12     -> FSCAL_R mask       (0x80f0000000000000)
  x13     -> E reg. clear mask  (0x00ffffffffffffff)
  x14     -> E reg. set mask    (0x3*00000000******)
  x15     -> E reg. set mask    (0x3*00000000******)
  x16-x23 -> VM registers "r0"-"r7"
  x24     -> iteration counter "ic"
  x25     -> VM registers "mx", "ma"
  x26     -> spAddr0
  x27     -> spAddr1
  x28-x31 -> temp/literals for IMUL_RCP (4x)

  (Note: We avoid using x4 because it breaks debugging with gdb.)

  f0-f7   -> VM registers "f0"-"f3"
  f8-f15  -> VM registers "e0"-"e3"
  f16-f23 -> VM registers "a0"-"a3"
  f24-f25 -> temp
  f26-f31 -> literals for IMUL_RCP (6x)

  Stack layout:
  ------------------------
  sp+
  0   -> return address
  8   -> register file ptr
  16  -> saved x3-x4
  32  -> saved x8-x9
  48  -> saved x18-x27
  128 -> saved f8-f9
  144 -> saved f18-f27
  224 -> caller stack
*/

DECL(randomx_riscv64_prologue):
    addi sp, sp, -224
    /* scratchpad pointer */
    mv x5, x12
    /* register file pointer */
    sd x10, 8(sp)
    /* callee saved registers */
    sd x3, 16(sp)
    sd x8, 32(sp)
    sd x9, 40(sp)
    sd x18, 48(sp)
    sd x19, 56(sp)
    sd x20, 64(sp)
    sd x21, 72(sp)
    sd x22, 80(sp)
    sd x23, 88(sp)
    sd x24, 96(sp)
    sd x25, 104(sp)
    sd x26, 112(sp)
    sd x27, 120(sp)
    fsd f8, 128(sp)
    fsd f9, 136(sp)
    fsd f18, 144(sp)
    fsd f19, 152(sp)
    fsd f20, 160(sp)
    fsd f21, 168(sp)
    fsd f22, 176(sp)
    fsd f23, 184(sp)
    fsd f24, 192(sp)
    fsd f25, 200(sp)
    fsd f26, 208(sp)
    fsd f27, 216(sp)
    /* iteration counter */
    mv x24, x13
    /* return address */
    sd x1, 0(sp)
    /* literal pool */
    lla x3, literal_pool
    /* load (ma, mx) */
    ld x25, 0(x11)
    /* dataset ptr */
    ld x6, 8(x11)
    /* load dataset mask */
    lwu x1, 92(x3)
    /* zero registers r0-r3, load a0-a1 */
    li x16, 0
    fld f16, 192(x10)
    li x17, 0
    fld f17, 200(x10)
    srli x7, x25, 32 /* x7 = ma */
    li x18, 0
    fld f18, 208(x10)
    mv x27, x7 /* x27 = ma */
    li x19, 0
    fld f19, 216(x10)
    /* set dataset read address */
    and x7, x7, x1
    add x7, x7, x6
    /* zero registers r4-r7, load a2-a3 */
    li x20, 0
    fld f20, 224(x10)
    li x21, 0
    fld f21, 232(x10)
    li x22, 0
    fld f22, 240(x10)
    li x23, 0
    fld f23, 248(x10)
    /* load L3 mask */
    lwu x1, 88(x3)
    /* load scratchpad masks */
    lwu x10, 80(x3)
    lwu x11, 84(x3)
    /* set spAddr0, spAddr1 */
    and x26, x25, x1
    and x27, x27, x1
    add x26, x26, x5
    add x27, x27, x5
    /* align L3 mask */
    addi x1, x1, 56
    /* FSCAL, E reg. masks */
    ld x12, 96(x3)
    ld x13, 104(x3)
    ld x14, 112(x3)
    ld x15, 120(x3)
    /* IMUL_RCP literals */
    fld f26, 176(x3)
    fld f27, 184(x3)
    fld f28, 192(x3)
    fld f29, 200(x3)
    fld f30, 208(x3)
    fld f31, 216(x3)

.balign 4
DECL(randomx_riscv64_loop_begin):
loop_begin:
    /* mix integer registers */
    ld x8, 0(x26)
    ld x9, 8(x26)
    ld x30, 16(x26)
    ld x31, 24(x26)
    xor x16, x16, x8
    ld x8, 32(x26)
    xor x17, x17, x9
    ld x9, 40(x26)
    xor x18, x18, x30
    ld x30, 48(x26)
    xor x19, x19, x31
    ld x31, 56(x26)
    xor x20, x20, x8
    lw x8, 0(x27)
    xor x21, x21, x9
    lw x9, 4(x27)
    xor x22, x22, x30
    lw x30, 8(x27)
    xor x23, x23, x31
    lw x31, 12(x27)
    /* load F registers */
    fcvt.d.w f0, x8
    lw x8, 16(x27)
    fcvt.d.w f1, x9
    lw x9, 20(x27)
    fcvt.d.w f2, x30
    lw x30, 24(x27)
    fcvt.d.w f3, x31
    lw x31, 28(x27)
    fcvt.d.w f4, x8
    lw x8, 32(x27)
    fcvt.d.w f5, x9
    lw x9, 36(x27)
    fcvt.d.w f6, x30
    lw x30, 40(x27)
    fcvt.d.w f7, x31
    lw x31, 44(x27)
    /* load E registers */
    fcvt.d.w f8, x8
    lw x8, 48(x27)
    fcvt.d.w f9, x9
    lw x9, 52(x27)
    fcvt.d.w f10, x30
    lw x30, 56(x27)
    fcvt.d.w f11, x31
    lw x31, 60(x27)
    fcvt.d.w f12, x8
    fmv.x.d x8, f8
    fcvt.d.w f13, x9
    fmv.x.d x9, f9
    fcvt.d.w f14, x30
    fmv.x.d x30, f10
    fcvt.d.w f15, x31
    fmv.x.d x31, f11
    and x8, x8, x13
    and x9, x9, x13
    or x8, x8, x14
    or x9, x9, x15
    and x30, x30, x13
    and x31, x31, x13
    or x30, x30, x14
    or x31, x31, x15
    fmv.d.x f8, x8
    fmv.d.x f9, x9
    fmv.d.x f10, x30
    fmv.d.x f11, x31
    fmv.x.d x8, f12
    fmv.x.d x9, f13
    fmv.x.d x30, f14
    fmv.x.d x31, f15
    and x8, x8, x13
    and x9, x9, x13
    or x8, x8, x14
    or x9, x9, x15
    fmv.d.x f12, x8
    fmv.d.x f13, x9
    and x30, x30, x13
    and x31, x31, x13
    or x30, x30, x14
    or x31, x31, x15
    fmv.d.x f14, x30
    fmv.d.x f15, x31
    /* reload clobbered IMUL_RCP regs */
    ld x28, 144(x3)
    ld x29, 152(x3)
    ld x30, 160(x3)
    ld x31, 168(x3)

DECL(randomx_riscv64_data_read):
    xor x8, x20, x22 /* JIT compiler will adjust the registers */
    /* load dataset mask */
    lwu x1, 92(x3)
    /* zero-extend x8 */
#ifdef __riscv_zba
    zext.w x8, x8
#else
    slli x8, x8, 32
    srli x8, x8, 32
#endif
    /* update "mx" */
    xor x25, x25, x8
    /* read dataset and update registers */
    ld x8, 0(x7)
    ld x9, 8(x7)
    ld x30, 16(x7)
    ld x31, 24(x7)
    xor x16, x16, x8
    ld x8, 32(x7)
    xor x17, x17, x9
    ld x9, 40(x7)
    xor x18, x18, x30
    ld x30, 48(x7)
    xor x19, x19, x31
    ld x31, 56(x7)
    xor x20, x20, x8
    /* calculate the next dataset address */
    and x7, x25, x1
    xor x21, x21, x9
    add x7, x7, x6
    xor x22, x22, x30
    /* prefetch - doesn't seem to have any effect */
    /* ld x0, 0(x7) */
    xor x23, x23, x31
    /* swap mx <-> ma */
#ifdef __riscv_zbb
    rori x25, x25, 32
#else
    srli x9, x25, 32
    slli x25, x25, 32
    or x25, x25, x9
#endif

DECL(randomx_riscv64_data_read_light):
    xor x8, x20, x22 /* JIT compiler will adjust the registers */
    /* load dataset offset */
    lui x9, 0x02000  /* JIT compiler will adjust the immediate */
    addi x9, x9, -64
    /* load dataset mask */
    lwu x1, 92(x3)
    /* swap mx <-> ma */
#ifdef __riscv_zbb
    rori x25, x25, 32
#else
    srli x31, x25, 32
    slli x25, x25, 32
    or x25, x25, x31
#endif
    slli x8, x8, 32
    /* update "mx" */
    xor x25, x25, x8
    /* the next dataset item */
    and x7, x25, x1
    srli x7, x7, 6
    add x7, x7, x9
DECL(randomx_riscv64_fix_loop_call):
    jal superscalar_hash /* JIT compiler will adjust the offset */
    xor x16, x16, x8
    xor x17, x17, x9
    xor x18, x18, x10
    xor x19, x19, x11
    xor x20, x20, x12
    xor x21, x21, x13
    xor x22, x22, x14
    xor x23, x23, x15
    /* restore clobbered registers */
    lwu x10, 80(x3)
    lwu x11, 84(x3)
    ld x12, 96(x3)
    ld x13, 104(x3)
    ld x14, 112(x3)
    ld x15, 120(x3)

DECL(randomx_riscv64_spad_store):
    /* store integer registers */
    sd x16, 0(x27)
    sd x17, 8(x27)
    sd x18, 16(x27)
    sd x19, 24(x27)
    sd x20, 32(x27)
    sd x21, 40(x27)
    sd x22, 48(x27)
    sd x23, 56(x27)
    /* XOR and store f0,e0 */
    fmv.x.d x8, f0
    fmv.x.d x9, f8
    fmv.x.d x30, f1
    fmv.x.d x31, f9
    xor x8, x8, x9
    xor x30, x30, x31
    sd x8, 0(x26)
    fmv.d.x f0, x8
    sd x30, 8(x26)
    fmv.d.x f1, x30
    /* XOR and store f1,e1 */
    fmv.x.d x8, f2
    fmv.x.d x9, f10
    fmv.x.d x30, f3
    fmv.x.d x31, f11
    xor x8, x8, x9
    xor x30, x30, x31
    sd x8, 16(x26)
    fmv.d.x f2, x8
    sd x30, 24(x26)
    fmv.d.x f3, x30
    /* XOR and store f2,e2 */
    fmv.x.d x8, f4
    fmv.x.d x9, f12
    fmv.x.d x30, f5
    fmv.x.d x31, f13
    xor x8, x8, x9
    xor x30, x30, x31
    sd x8, 32(x26)
    fmv.d.x f4, x8
    sd x30, 40(x26)
    fmv.d.x f5, x30
    /* XOR and store f3,e3 */
    fmv.x.d x8, f6
    fmv.x.d x9, f14
    fmv.x.d x30, f7
    fmv.x.d x31, f15
    xor x8, x8, x9
    xor x30, x30, x31
    sd x8, 48(x26)
    fmv.d.x f6, x8
    sd x30, 56(x26)
    fmv.d.x f7, x30

DECL(randomx_riscv64_spad_store_hardaes):
    nop /* not implemented */

DECL(randomx_riscv64_spad_store_softaes):
    /* store integer registers */
    sd x16, 0(x27)
    sd x17, 8(x27)
    sd x18, 16(x27)
    sd x19, 24(x27)
    sd x20, 32(x27)
    sd x21, 40(x27)
    sd x22, 48(x27)
    sd x23, 56(x27)
    /* process f0 with 4 AES rounds */
    fmv.x.d x8, f8
    fmv.x.d x10, f9
    fmv.x.d x30, f0
    fmv.x.d x31, f1
    jal softaes_enc
    fmv.x.d x8, f10
    fmv.x.d x10, f11
    jal softaes_enc
    fmv.x.d x8, f12
    fmv.x.d x10, f13
    jal softaes_enc
    fmv.x.d x8, f14
    fmv.x.d x10, f15
    jal softaes_enc
    sd x30, 0(x26)
    fmv.d.x f0, x30
    sd x31, 8(x26)
    fmv.d.x f1, x31
    /* process f1 with 4 AES rounds */
    fmv.x.d x8, f8
    fmv.x.d x10, f9
    fmv.x.d x30, f2
    fmv.x.d x31, f3
    jal softaes_dec
    fmv.x.d x8, f10
    fmv.x.d x10, f11
    jal softaes_dec
    fmv.x.d x8, f12
    fmv.x.d x10, f13
    jal softaes_dec
    fmv.x.d x8, f14
    fmv.x.d x10, f15
    jal softaes_dec
    sd x30, 16(x26)
    fmv.d.x f2, x30
    sd x31, 24(x26)
    fmv.d.x f3, x31
    /* process f2 with 4 AES rounds */
    fmv.x.d x8, f8
    fmv.x.d x10, f9
    fmv.x.d x30, f4
    fmv.x.d x31, f5
    jal softaes_enc
    fmv.x.d x8, f10
    fmv.x.d x10, f11
    jal softaes_enc
    fmv.x.d x8, f12
    fmv.x.d x10, f13
    jal softaes_enc
    fmv.x.d x8, f14
    fmv.x.d x10, f15
    jal softaes_enc
    sd x30, 32(x26)
    fmv.d.x f4, x30
    sd x31, 40(x26)
    fmv.d.x f5, x31
    /* process f3 with 4 AES rounds */
    fmv.x.d x8, f8
    fmv.x.d x10, f9
    fmv.x.d x30, f6
    fmv.x.d x31, f7
    jal softaes_dec
    fmv.x.d x8, f10
    fmv.x.d x10, f11
    jal softaes_dec
    fmv.x.d x8, f12
    fmv.x.d x10, f13
    jal softaes_dec
    fmv.x.d x8, f14
    fmv.x.d x10, f15
    jal softaes_dec
    sd x30, 48(x26)
    fmv.d.x f6, x30
    sd x31, 56(x26)
    fmv.d.x f7, x31
    /* restore clobbered registers */
    lwu x10, 80(x3)
    lwu x11, 84(x3)
    ld x12, 96(x3)
    ld x13, 104(x3)
    ld x14, 112(x3)
    ld x15, 120(x3)

DECL(randomx_riscv64_loop_end):
    xor x26, x16, x18 /* JIT compiler will adjust the registers */
    /* load L3 mask */
    lwu x1, 88(x3)
    addi x24, x24, -1
    srli x27, x26, 32
    /* set spAddr0, spAddr1 */
    and x26, x26, x1
    and x27, x27, x1
    add x26, x26, x5
    add x27, x27, x5
    /* align L3 mask */
    addi x1, x1, 56
    /* conditional branch doesn't have sufficient range */
    j condition_check
DECL(randomx_riscv64_fix_continue_loop):
continue_loop:
    .word 0 /* JIT compiler will write a jump to loop_begin  */
condition_check:
    bnez x24, continue_loop

DECL(randomx_riscv64_epilogue):
    /* restore callee saved registers */
    ld x10, 8(sp)
    ld x1, 0(sp)
    ld x3, 16(sp)
    ld x8, 32(sp)
    ld x9, 40(sp)
    ld x24, 96(sp)
    ld x25, 104(sp)
    ld x26, 112(sp)
    ld x27, 120(sp)
    fld f18, 144(sp)
    fld f19, 152(sp)
    fld f20, 160(sp)
    fld f21, 168(sp)
    fld f22, 176(sp)
    fld f23, 184(sp)
    fld f24, 192(sp)
    fld f25, 200(sp)
    fld f26, 208(sp)
    fld f27, 216(sp)
    /* save VM registers */
    sd x16, 0(x10)
    sd x17, 8(x10)
    sd x18, 16(x10)
    sd x19, 24(x10)
    sd x20, 32(x10)
    sd x21, 40(x10)
    sd x22, 48(x10)
    sd x23, 56(x10)
    fsd f0, 64(x10)
    fsd f1, 72(x10)
    fsd f2, 80(x10)
    fsd f3, 88(x10)
    fsd f4, 96(x10)
    fsd f5, 104(x10)
    fsd f6, 112(x10)
    fsd f7, 120(x10)
    fsd f8, 128(x10)
    fsd f9, 136(x10)
    fsd f10, 144(x10)
    fsd f11, 152(x10)
    fsd f12, 160(x10)
    fsd f13, 168(x10)
    fsd f14, 176(x10)
    fsd f15, 184(x10)
    /* restore callee saved registers */
    ld x18, 48(sp)
    ld x19, 56(sp)
    ld x20, 64(sp)
    ld x21, 72(sp)
    ld x22, 80(sp)
    ld x23, 88(sp)
    fld f8, 128(sp)
    fld f9, 136(sp)
    /* restore stack pointer */
    addi sp, sp, 224
    /* return */
    ret

/*
    Soft AES subroutines
        in:
                  x3 = literal pool
             x8, x10 = round key
            x30, x31 = plaintext
        out:
            x30, x31 = ciphertext
        clobbers:
             x8-x11 (limbs)
            x12-x13 (LUTs)
            x14-x15 (temp)
*/
DECL(randomx_riscv64_softaes):
softaes_enc:
    /* enc. lookup table */
    ld x13, 128(x3)

    /* load the round key into x8, x9, x10, x11 */
    srli x9, x8, 32
    srli x11, x10, 32
#ifdef __riscv_zba
    zext.w x8, x8
    zext.w x10, x10
#else
    slli x8, x8, 32
    slli x10, x10, 32
    srli x8, x8, 32
    srli x10, x10, 32
#endif

    /* byte 0 */
    andi x14, x30, 255
    srli x30, x30, 8
    addi x12, x13, -2048
#ifdef __riscv_zba
    sh2add x14, x14, x13
#else
    slli x14, x14, 2
    add x14, x14, x13
#endif
    lwu x14, -2048(x14)

    /* byte 1 */
    andi x15, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x15, x15, x12
#else
    slli x15, x15, 2
    add x15, x15, x12
#endif
    lwu x15, 1024(x15)
    xor x8, x8, x14

    /* byte 2 */
    andi x14, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x14, x14, x13
#else
    slli x14, x14, 2
    add x14, x14, x13
#endif
    lwu x14, 0(x14)
    xor x11, x11, x15

    /* byte 3 */
    andi x15, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x15, x15, x13
#else
    slli x15, x15, 2
    add x15, x15, x13
#endif
    lwu x15, 1024(x15)
    xor x10, x10, x14

    /* byte 4 */
    andi x14, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x14, x14, x12
#else
    slli x14, x14, 2
    add x14, x14, x12
#endif
    lwu x14, 0(x14)
    xor x9, x9, x15

    /* byte 5 */
    andi x15, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x15, x15, x12
#else
    slli x15, x15, 2
    add x15, x15, x12
#endif
    lwu x15, 1024(x15)
    xor x9, x9, x14

    /* byte 6 */
    andi x14, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x14, x14, x13
#else
    slli x14, x14, 2
    add x14, x14, x13
#endif
    lwu x14, 0(x14)
    xor x8, x8, x15

    /* byte 7 */
    andi x15, x30, 255
#ifdef __riscv_zba
    sh2add x15, x15, x13
#else
    slli x15, x15, 2
    add x15, x15, x13
#endif
    lwu x15, 1024(x15)
    xor x11, x11, x14

    /* byte 8 */
    andi x14, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x14, x14, x12
#else
    slli x14, x14, 2
    add x14, x14, x12
#endif
    lwu x14, 0(x14)
    xor x10, x10, x15

    /* byte 9 */
    andi x15, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x15, x15, x12
#else
    slli x15, x15, 2
    add x15, x15, x12
#endif
    lwu x15, 1024(x15)
    xor x10, x10, x14

    /* byte 10 */
    andi x14, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x14, x14, x13
#else
    slli x14, x14, 2
    add x14, x14, x13
#endif
    lwu x14, 0(x14)
    xor x9, x9, x15

    /* byte 11 */
    andi x15, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x15, x15, x13
#else
    slli x15, x15, 2
    add x15, x15, x13
#endif
    lwu x15, 1024(x15)
    xor x8, x8, x14

    /* byte 12 */
    andi x14, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x14, x14, x12
#else
    slli x14, x14, 2
    add x14, x14, x12
#endif
    lwu x14, 0(x14)
    xor x11, x11, x15

    /* byte 13 */
    andi x15, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x15, x15, x12
#else
    slli x15, x15, 2
    add x15, x15, x12
#endif
    lwu x15, 1024(x15)
    xor x11, x11, x14

    /* byte 14 */
    andi x14, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x14, x14, x13
#else
    slli x14, x14, 2
    add x14, x14, x13
#endif
    lwu x14, 0(x14)
    xor x10, x10, x15

    /* byte 15 */
    andi x15, x31, 255
#ifdef __riscv_zba
    sh2add x15, x15, x13
#else
    slli x15, x15, 2
    add x15, x15, x13
#endif
    lwu x15, 1024(x15)
    xor x9, x9, x14

    slli x11, x11, 32
    slli x9, x9, 32
    or x30, x8, x9
    or x31, x10, x11
    xor x30, x30, x15

    ret

softaes_dec:
    /* dec. lookup table */
    ld x13, 136(x3)

    /* load the round key into x8, x9, x10, x11 */
    srli x9, x8, 32
    srli x11, x10, 32
#ifdef __riscv_zba
    zext.w x8, x8
    zext.w x10, x10
#else
    slli x8, x8, 32
    slli x10, x10, 32
    srli x8, x8, 32
    srli x10, x10, 32
#endif

    /* byte 0 */
    andi x14, x30, 255
    srli x30, x30, 8
    addi x12, x13, -2048
#ifdef __riscv_zba
    sh2add x14, x14, x13
#else
    slli x14, x14, 2
    add x14, x14, x13
#endif
    lwu x14, -2048(x14)

    /* byte 1 */
    andi x15, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x15, x15, x12
#else
    slli x15, x15, 2
    add x15, x15, x12
#endif
    lwu x15, 1024(x15)
    xor x8, x8, x14

    /* byte 2 */
    andi x14, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x14, x14, x13
#else
    slli x14, x14, 2
    add x14, x14, x13
#endif
    lwu x14, 0(x14)
    xor x9, x9, x15

    /* byte 3 */
    andi x15, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x15, x15, x13
#else
    slli x15, x15, 2
    add x15, x15, x13
#endif
    lwu x15, 1024(x15)
    xor x10, x10, x14

    /* byte 4 */
    andi x14, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x14, x14, x12
#else
    slli x14, x14, 2
    add x14, x14, x12
#endif
    lwu x14, 0(x14)
    xor x11, x11, x15

    /* byte 5 */
    andi x15, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x15, x15, x12
#else
    slli x15, x15, 2
    add x15, x15, x12
#endif
    lwu x15, 1024(x15)
    xor x9, x9, x14

    /* byte 6 */
    andi x14, x30, 255
    srli x30, x30, 8
#ifdef __riscv_zba
    sh2add x14, x14, x13
#else
    slli x14, x14, 2
    add x14, x14, x13
#endif
    lwu x14, 0(x14)
    xor x10, x10, x15

    /* byte 7 */
    andi x15, x30, 255
#ifdef __riscv_zba
    sh2add x15, x15, x13
#else
    slli x15, x15, 2
    add x15, x15, x13
#endif
    lwu x15, 1024(x15)
    xor x11, x11, x14

    /* byte 8 */
    andi x14, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x14, x14, x12
#else
    slli x14, x14, 2
    add x14, x14, x12
#endif
    lwu x14, 0(x14)
    xor x8, x8, x15

    /* byte 9 */
    andi x15, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x15, x15, x12
#else
    slli x15, x15, 2
    add x15, x15, x12
#endif
    lwu x15, 1024(x15)
    xor x10, x10, x14

    /* byte 10 */
    andi x14, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x14, x14, x13
#else
    slli x14, x14, 2
    add x14, x14, x13
#endif
    lwu x14, 0(x14)
    xor x11, x11, x15

    /* byte 11 */
    andi x15, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x15, x15, x13
#else
    slli x15, x15, 2
    add x15, x15, x13
#endif
    lwu x15, 1024(x15)
    xor x8, x8, x14

    /* byte 12 */
    andi x14, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x14, x14, x12
#else
    slli x14, x14, 2
    add x14, x14, x12
#endif
    lwu x14, 0(x14)
    xor x9, x9, x15

    /* byte 13 */
    andi x15, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x15, x15, x12
#else
    slli x15, x15, 2
    add x15, x15, x12
#endif
    lwu x15, 1024(x15)
    xor x11, x11, x14

    /* byte 14 */
    andi x14, x31, 255
    srli x31, x31, 8
#ifdef __riscv_zba
    sh2add x14, x14, x13
#else
    slli x14, x14, 2
    add x14, x14, x13
#endif
    lwu x14, 0(x14)
    xor x8, x8, x15

    /* byte 15 */
    andi x15, x31, 255
#ifdef __riscv_zba
    sh2add x15, x15, x13
#else
    slli x15, x15, 2
    add x15, x15, x13
#endif
    lwu x15, 1024(x15)
    xor x9, x9, x14

    slli x11, x11, 32
    slli x9, x9, 32
    or x30, x8, x9
    or x31, x10, x11
    xor x31, x31, x15

    ret

DECL(randomx_riscv64_program_end):
    nop


/* literal pool for SuperscalarHash */
    /* space for remaining IMUL_RCP literals */
ssh_literal_pool:
    /* space for 256 IMUL_RCP literals */
    .fill 256,8,0

/*
    SuperscalarHash subroutine
        in:
            x3 = literal pool
            x6 = cache
            x7 = itemNumber
        out:
            x8-x15 = 64-byte hash
        clobbers:
            x7, x28-x31
*/
DECL(randomx_riscv64_ssh_init):
superscalar_hash:
    ld x30, 0(x3) /* superscalarMul0 */
    addi x8, x7, 1
    ld x9, 8(x3)
    li x31, RANDOMX_CACHE_MASK
    ld x10, 16(x3)
    ld x11, 24(x3)
    mul x8, x8, x30
    ld x12, 32(x3)
    ld x13, 40(x3)
    lla x30, ssh_literal_pool
    ld x14, 48(x3)
    and x7, x7, x31
    ld x15, 56(x3)
    slli x7, x7, 6
    xor x9, x9, x8
    add x7, x7, x6
    xor x10, x10, x8
    /* load the first IMUL_RCP literal */
    ld x31, 2040(x30)
    xor x11, x11, x8
    xor x12, x12, x8
    xor x13, x13, x8
    xor x14, x14, x8
    xor x15, x15, x8

DECL(randomx_riscv64_ssh_load):
    ld x28, 0(x7)
    ld x29, 8(x7)
    xor x8, x8, x28
    ld x28, 16(x7)
    xor x9, x9, x29
    ld x29, 24(x7)
    xor x10, x10, x28
    ld x28, 32(x7)
    xor x11, x11, x29
    ld x29, 40(x7)
    xor x12, x12, x28
    ld x28, 48(x7)
    xor x13, x13, x29
    ld x29, 56(x7)
    xor x14, x14, x28
    li x7, RANDOMX_CACHE_MASK
    xor x15, x15, x29

DECL(randomx_riscv64_ssh_prefetch):
    and x7, x8, x7   /* JIT compiler will adjust the register */
    slli x7, x7, 6
    add x7, x7, x6
    /* prefetch - doesn't seem to have any effect */
    /* ld x0, 0(x7) */

DECL(randomx_riscv64_ssh_end):
    nop
