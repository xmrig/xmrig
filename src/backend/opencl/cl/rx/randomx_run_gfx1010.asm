/*
Copyright (c) 2019-2020 SChernykh

This file is part of RandomX OpenCL.

RandomX OpenCL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX OpenCL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX OpenCL. If not, see <http://www.gnu.org/licenses/>.
*/

.rocm
.gpu GFX1010
.arch_minor 1
.arch_stepping 0
.eflags 53
.llvm10binfmt
.metadatav3
.md_version 1, 0
.globaldata
    .fill 64, 1, 0
.kernel randomx_run
    .config
        .dims x
        .sgprsnum 96
        .vgprsnum 128
        .shared_vgprs 0
        .dx10clamp
        .ieeemode
        .floatmode 0xf0
        .priority 0
        .exceptions 0
        .userdatanum 6

        # https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-compute-pgm-rsrc1-gfx6-gfx10-table
        # https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-compute-pgm-rsrc2-gfx6-gfx10-table
        # https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-compute-pgm-rsrc3-gfx10-table
        .pgmrsrc1 0x40af0105
        .pgmrsrc2 0x0000008c
        .pgmrsrc3 0x00000000

        .group_segment_fixed_size 256
        .private_segment_fixed_size 0
        .kernel_code_entry_offset 0x10c0
        .use_private_segment_buffer
        .use_kernarg_segment_ptr
        .use_wave32
    .config
        .md_symname "randomx_run.kd"
        .md_language "OpenCL C", 1, 2
        .reqd_work_group_size 32, 1, 1
        .md_kernarg_segment_size 104
        .md_kernarg_segment_align 8
        .md_group_segment_fixed_size 256
        .md_private_segment_fixed_size 0
        .md_wavefront_size 32
        .md_sgprsnum 96
        .md_vgprsnum 128
        .spilledsgprs 0
        .spilledvgprs 0
        .max_flat_work_group_size 32
        .arg dataset, "uchar*", 8, 0, globalbuf, u8, global, default const
        .arg scratchpad, "uchar*", 8, 8, globalbuf, u8, global, default
        .arg registers, "ulong*", 8, 16, globalbuf, u64, global, default
        .arg rounding_modes, "uint*", 8, 24, globalbuf, u32, global, default
        .arg programs, "uint*", 8, 32, globalbuf, u32, global, default
        .arg batch_size, "uint", 4, 40, value, u32
        .arg rx_parameters, "uint", 4, 44, value, u32
        .arg , "", 8, 48, gox, i64
        .arg , "", 8, 56, goy, i64
        .arg , "", 8, 64, goz, i64
        .arg , "", 8, 72, none, i8
        .arg , "", 8, 80, none, i8
        .arg , "", 8, 88, none, i8
        .arg , "", 8, 96, multigridsyncarg, i8
.text
randomx_run:
    # clear all caches
    s_dcache_wb
    s_waitcnt       vmcnt(0) & lgkmcnt(0)
    s_waitcnt_vscnt null, 0x0
    s_icache_inv
    s_branch begin

    # pgmrsrc2 = 0x0000008c, bits 1:5 = 6, so first 6 SGPRs (s0-s7) contain user data
    # s6 contains group id
    # v0 contains local id
begin:
    # s[0:1] - pointer to registers
    # s[2:3] - pointer to rounding modes
    s_load_dwordx4  s[0:3], s[4:5], 0x10

    # s[8:9] - group_id*group_size
    s_mov_b32       s9, 0
    s_lshl_b32      s8, s6, 5

    # v0 - local id (sub)
    # v39 - R[sub]
    v_lshlrev_b32   v39, 3, v0

    s_mov_b32       s12, s7

    # vcc_lo = "if (sub < 8)"
    v_cmp_gt_u32    vcc_lo, 8, v0

    s_waitcnt       lgkmcnt(0)

    # load rounding mode
    s_lshl_b32      s16, s6, 2
    s_add_u32       s64, s2, s16
    s_addc_u32      s65, s3, 0
    v_mov_b32       v1, 0
    global_load_dword v1, v1, s[64:65]
    s_waitcnt       vmcnt(0)
    v_readlane_b32  s66, v1, 0
    s_setreg_b32    hwreg(mode, 2, 2), s66
    s_mov_b32       s67, 0

    # ((__local ulong*) R)[sub] = ((__global ulong*) registers)[sub];
    s_lshl_b64      s[2:3], s[8:9], 3
    s_mov_b32       s32, s12
    s_add_u32       s0, s0, s2
    s_addc_u32      s1, s1, s3
    v_add_co_u32    v1, s0, s0, v39
    v_add_co_ci_u32 v2, s0, s1, 0, s0
    global_load_dwordx2 v[4:5], v[1:2], off
    s_waitcnt       vmcnt(0)
    ds_write_b64    v39, v[4:5]
    s_waitcnt       vmcnt(0) & lgkmcnt(0)
    s_waitcnt_vscnt null, 0x0

    # "if (sub >= 8) return"
    s_and_saveexec_b32 s0, vcc_lo
    s_cbranch_execz program_end

    # s[8:9] - pointer to dataset
    # s[10:11] - pointer to scratchpads
    # s[0:1] - pointer to programs
    s_load_dwordx4  s[8:11], s[4:5], 0x0
    s_load_dwordx2  s[0:1], s[4:5], 0x20

    # rx_parameters
    s_load_dword    s20, s[4:5], 0x2c

    v_mov_b32       v5, 0
    v_mov_b32       v10, 0
    s_waitcnt_vscnt null, 0x0
    ds_read_b64     v[8:9], v39
    v_cmp_gt_u32    vcc_lo, 4, v0
    v_lshlrev_b32   v0, 3, v0
    ds_read2_b64    v[25:28], v5 offset0:16 offset1:17
    ds_read_b32     v11, v5 offset:152
    ds_read_b64     v[35:36], v5 offset:168
    ds_read2_b64    v[20:23], v5 offset0:18 offset1:20
    v_cndmask_b32   v4, 0xffffff, -1, vcc_lo
    v_add_nc_u32    v5, v39, v0
    s_waitcnt       lgkmcnt(0)
    v_mov_b32       v13, s11
    v_mov_b32       v7, s1
    v_mov_b32       v6, s0

    # Scratchpad L1 size
    s_bfe_u32       s21, s20, 0x050000
    s_lshl_b32      s21, 1, s21

    # Scratchpad L2 size
    s_bfe_u32       s22, s20, 0x050005
    s_lshl_b32      s22, 1, s22

    # Scratchpad L3 size
    s_bfe_u32       s0, s20, 0x05000A
    s_lshl_b32      s23, 1, s0

    # program iterations
    s_bfe_u32       s24, s20, 0x04000F
    s_lshl_b32      s24, 1, s24

    v_mov_b32       v12, s10
    v_mad_u64_u32   v[6:7], s2, 10048, s6, v[6:7]

    # s[4:5] - pointer to current program
    v_readlane_b32  s4, v6, 0
    v_readlane_b32  s5, v7, 0

    s_lshl_b32      s2, 1, s0
    v_add_co_u32    v14, s0, s8, v11
    v_cndmask_b32   v34, v36, 0, vcc_lo
    v_cndmask_b32   v24, v23, 0, vcc_lo
    v_cndmask_b32   v3, v22, 0, vcc_lo
    s_add_i32       s3, s2, 64
    v_add_co_ci_u32 v29, s0, s9, v10, s0
    v_cndmask_b32   v35, v35, 0, vcc_lo
    v_add_co_u32    v22, vcc_lo, v14, v0

    # v[12:13] - pointer to current scratchpad
    v_mad_u64_u32   v[12:13], s2, s3, s6, v[12:13]
    v_mov_b32       v10, v26
    v_mov_b32       v11, v25
    v_lshlrev_b32   v36, 3, v27
    v_lshlrev_b32   v37, 3, v28
    v_lshlrev_b32   v20, 3, v20
    v_lshlrev_b32   v21, 3, v21
    v_add_co_ci_u32 v23, vcc_lo, 0, v29, vcc_lo

    # rename registers
    # v6 - R[sub]
    v_mov_b32 v6, v39

    # loop counter
    s_sub_u32       s2, s24, 1

    # used in IXOR_R instruction
    s_mov_b32       s63, -1

    # used in CBRANCH instruction
    s_mov_b32       s70, (0xFF << 8)
    s_mov_b32       s71, (0xFF << 9)
    s_mov_b32       s72, (0xFF << 10)
    s_mov_b32       s73, (0xFF << 11)
    s_mov_b32       s74, (0xFF << 12)
    s_mov_b32       s75, (0xFF << 13)
    s_mov_b32       s76, (0xFF << 14)
    s_mov_b32       s77, (0xFF << 15)
    s_mov_b32       s78, (0xFF << 16)
    s_mov_b32       s79, (0xFF << 17)
    s_mov_b32       s80, (0xFF << 18)
    s_mov_b32       s81, (0xFF << 19)
    s_mov_b32       s82, (0xFF << 20)
    s_mov_b32       s83, (0xFF << 21)
    s_mov_b32       s84, (0xFF << 22)
    s_mov_b32       s85, (0xFF << 23)

    # ScratchpadL3Mask64
    s_sub_u32       s86, s23, 64

    # Scratchpad masks for scratchpads
    v_sub_nc_u32    v38, s21, 8
    v_sub_nc_u32    v39, s22, 8
    v_sub_nc_u32    v50, s23, 8

    # mask for FSCAL_R
    v_mov_b32       v51, 0x80F00000

    # load scratchpad base address
    v_readlane_b32  s0, v12, 0
    v_readlane_b32  s1, v13, 0

    # v41, v44 = 0
    v_mov_b32       v41, 0
    v_mov_b32       v44, 0

    # v41 = 0 on lane 0, set it to 8 on lane 1
    # v44 = 0 on lane 0, set it to 4 on lane 1
    s_mov_b64       exec, 2
    v_mov_b32       v41, 8
    v_mov_b32       v44, 4

    # load group A registers
    # Read low 8 bytes into lane 0 and high 8 bytes into lane 1
    s_mov_b64       exec, 3
    ds_read2_b64    v[52:55], v41 offset0:24 offset1:26
    ds_read2_b64    v[56:59], v41 offset0:28 offset1:30

    # xmantissaMask
    v_mov_b32       v77, (1 << 24) - 1

    # xexponentMask
    ds_read_b64     v[78:79], v41 offset:160

    # Restore execution mask
    s_mov_b64       exec, 255

    # sign mask (used in FSQRT_R)
    v_mov_b32       v82, 0x80000000

    # used in FSQRT_R to check for "positive normal value" (v_cmpx_class_f64)
    s_mov_b32       s68, 256
    s_mov_b32       s69, 0

    # High 32 bits of "1.0" constant (used in FDIV_M)
    v_mov_b32       v83, (1023 << 20)

    # Used to multiply FP64 values by 0.5
    v_mov_b32       v84, (1 << 20)

    s_getpc_b64 s[14:15]
cur_addr:

    # get addresses of FSQRT_R subroutines
    s_add_u32       s40, s14, fsqrt_r_sub0 - cur_addr
    s_addc_u32      s41, s15, 0
    s_add_u32       s42, s14, fsqrt_r_sub1 - cur_addr
    s_addc_u32      s43, s15, 0
    s_add_u32       s44, s14, fsqrt_r_sub2 - cur_addr
    s_addc_u32      s45, s15, 0
    s_add_u32       s46, s14, fsqrt_r_sub3 - cur_addr
    s_addc_u32      s47, s15, 0

    # get addresses of FDIV_M subroutines
    s_add_u32       s48, s14, fdiv_m_sub0 - cur_addr
    s_addc_u32      s49, s15, 0
    s_add_u32       s50, s14, fdiv_m_sub1 - cur_addr
    s_addc_u32      s51, s15, 0
    s_add_u32       s52, s14, fdiv_m_sub2 - cur_addr
    s_addc_u32      s53, s15, 0
    s_add_u32       s54, s14, fdiv_m_sub3 - cur_addr
    s_addc_u32      s55, s15, 0

    # get address for ISMULH_R subroutine
    s_add_u32       s56, s14, ismulh_r_sub - cur_addr
    s_addc_u32      s57, s15, 0

    # get address for IMULH_R subroutine
    s_add_u32       s58, s14, imulh_r_sub - cur_addr
    s_addc_u32      s59, s15, 0

/*
    used: v0-v6, v8-v37
    not used: v7
*/
main_loop:
    s_waitcnt_vscnt null, 0x0

    # v[27:28] = R[readReg0]
    # v[29:30] = R[readReg1]
    ds_read_b64     v[27:28], v37
    ds_read_b64     v[29:30], v36
    s_waitcnt       lgkmcnt(0)

    # R[readReg0] ^ R[readReg0] (high 32 bits)
    v_xor_b32       v28, v28, v30

    # spAddr1
    v_xor_b32       v25, v28, v25
    v_and_b32       v25, s86, v25
    v_add_nc_u32    v25, v25, v0

    v_add_co_u32    v16, vcc_lo, s0, v25

    # R[readReg0] ^ R[readReg0] (low 32 bits)
    v_xor_b32       v25, v27, v29

    v_mov_b32       v29, v11
    v_add_co_ci_u32 v17, vcc_lo, 0, s1, vcc_lo
    v_xor_b32       v25, v25, v26

    # load from spAddr1
    global_load_dwordx2 v[27:28], v[16:17], off

    # spAddr0
    v_and_b32       v25, s86, v25
    v_add_nc_u32    v25, v25, v0

    v_add_co_u32    v31, vcc_lo, s0, v25
    v_add_co_ci_u32 v32, vcc_lo, 0, s1, vcc_lo
    v_add_co_u32    v29, vcc_lo, v22, v29

    # load from spAddr0
    global_load_dwordx2 v[25:26], v[31:32], off

    v_add_co_ci_u32 v30, vcc_lo, 0, v23, vcc_lo
    v_mov_b32       v33, v11
    s_and_b32       vcc_lo, exec_lo, vcc_lo
    s_waitcnt       vmcnt(1)
    v_cvt_f64_i32   v[14:15], v28
    v_cvt_f64_i32   v[12:13], v27
    v_or_b32        v14, v14, v35
    s_waitcnt       vmcnt(0)

    # R[sub] ^= *p0;
    v_xor_b32       v8, v25, v8
    v_xor_b32       v9, v26, v9

    v_and_b32       v26, v4, v15

    v_and_b32       v19, v4, v13
    v_or_b32        v15, v26, v34
    v_or_b32        v18, v12, v3
    v_mov_b32       v26, 0
    v_or_b32        v19, v19, v24
    v_mov_b32       v25, v26
    ds_write2_b64   v5, v[18:19], v[14:15] offset0:8 offset1:9

    # load from dataset
    global_load_dwordx2 v[18:19], v[29:30], off

    # load group F,E registers
    # Read low 8 bytes into lane 0 and high 8 bytes into lane 1
    s_mov_b64       exec, 3
    s_waitcnt       lgkmcnt(0)
    ds_read2_b64    v[60:63], v41 offset0:8 offset1:10
    ds_read2_b64    v[64:67], v41 offset0:12 offset1:14
    ds_read2_b64    v[68:71], v41 offset0:16 offset1:18
    ds_read2_b64    v[72:75], v41 offset0:20 offset1:22

    # load VM integer registers
    v_readlane_b32  s16, v8, 0
    v_readlane_b32  s17, v9, 0
    v_readlane_b32  s18, v8, 1
    v_readlane_b32  s19, v9, 1
    v_readlane_b32  s20, v8, 2
    v_readlane_b32  s21, v9, 2
    v_readlane_b32  s22, v8, 3
    v_readlane_b32  s23, v9, 3
    v_readlane_b32  s24, v8, 4
    v_readlane_b32  s25, v9, 4
    v_readlane_b32  s26, v8, 5
    v_readlane_b32  s27, v9, 5
    v_readlane_b32  s28, v8, 6
    v_readlane_b32  s29, v9, 6
    v_readlane_b32  s30, v8, 7
    v_readlane_b32  s31, v9, 7

    s_waitcnt       lgkmcnt(0)

    # Use only first 2 lanes for the program
    s_mov_b64       exec, 3

    # call JIT code
    s_swappc_b64    s[12:13], s[4:5]

    # Write out group F,E registers
    # Write low 8 bytes from lane 0 and high 8 bytes from lane 1
    ds_write2_b64   v41, v[60:61], v[62:63] offset0:8 offset1:10
    ds_write2_b64   v41, v[64:65], v[66:67] offset0:12 offset1:14
    ds_write2_b64   v41, v[68:69], v[70:71] offset0:16 offset1:18
    ds_write2_b64   v41, v[72:73], v[74:75] offset0:20 offset1:22

    # store VM integer registers
    v_writelane_b32 v8, s16, 0
    v_writelane_b32 v9, s17, 0
    v_writelane_b32 v8, s18, 1
    v_writelane_b32 v9, s19, 1
    v_writelane_b32 v8, s20, 2
    v_writelane_b32 v9, s21, 2
    v_writelane_b32 v8, s22, 3
    v_writelane_b32 v9, s23, 3
    v_writelane_b32 v8, s24, 4
    v_writelane_b32 v9, s25, 4
    v_writelane_b32 v8, s26, 5
    v_writelane_b32 v9, s27, 5
    v_writelane_b32 v8, s28, 6
    v_writelane_b32 v9, s29, 6
    v_writelane_b32 v8, s30, 7
    v_writelane_b32 v9, s31, 7

    # Turn back on 8 execution lanes
    s_mov_b64       exec, 255

    # Write out VM integer registers
    ds_write_b64    v6, v[8:9]
    s_waitcnt       lgkmcnt(0)

    # R[readReg2], R[readReg3]
    ds_read_b32     v11, v21
    ds_read_b32     v27, v20
    s_waitcnt       lgkmcnt(0)

    # mx ^= R[readReg2] ^ R[readReg3];
    v_xor_b32       v11, v11, v27
    v_xor_b32       v10, v10, v11

    # v[27:28] = R[sub]
    # v[29:30] = F[sub]
    ds_read2_b64    v[27:30], v6 offset1:8

    # mx &= CacheLineAlignMask;
    v_and_b32       v11, 0x7fffffc0, v10
    v_mov_b32       v10, v33
    s_waitcnt       lgkmcnt(0)

    # const ulong next_r = R[sub] ^ data;
    s_waitcnt       lgkmcnt(0)
    v_xor_b32       v8, v27, v18
    v_xor_b32       v9, v28, v19

    # *p1 = next_r;
    global_store_dwordx2 v[16:17], v[8:9], off

    # v[27:28] = E[sub]
    ds_read_b64     v[27:28], v6 offset:128

    # R[sub] = next_r;
    ds_write_b64    v6, v[8:9]
    s_waitcnt       lgkmcnt(1)

    # *p0 = as_ulong(F[sub]) ^ as_ulong(E[sub]);
    v_xor_b32       v29, v27, v29
    v_xor_b32       v30, v28, v30
    global_store_dwordx2 v[31:32], v[29:30], off

    s_sub_u32       s2, s2, 1
    s_cbranch_scc0  main_loop
main_loop_end:

    global_store_dwordx2 v[1:2], v[8:9], off
    global_store_dwordx2 v[1:2], v[29:30], off inst_offset:64
    global_store_dwordx2 v[1:2], v[27:28], off inst_offset:128

    # store rounding mode
    v_mov_b32       v0, 0
    v_mov_b32       v1, s66
    global_store_dword v0, v1, s[64:65]

program_end:
    s_endpgm

fsqrt_r_sub0:
    s_setreg_b32    hwreg(mode, 2, 2), s67
    v_rsq_f64       v[28:29], v[68:69]

    # Improve initial approximation (can be skipped)
    #v_mul_f64       v[42:43], v[28:29], v[68:69]
    #v_mul_f64       v[48:49], v[28:29], -0.5
    #v_fma_f64       v[48:49], v[48:49], v[42:43], 0.5
    #v_fma_f64       v[28:29], v[28:29], v[48:49], v[28:29]

    v_mul_f64       v[42:43], v[28:29], v[68:69]
    v_mov_b32       v48, v28
    v_sub_nc_u32    v49, v29, v84
    v_mov_b32       v46, v28
    v_xor_b32       v47, v49, v82
    v_fma_f64       v[46:47], v[46:47], v[42:43], 0.5
    v_fma_f64       v[42:43], v[42:43], v[46:47], v[42:43]
    v_fma_f64       v[48:49], v[48:49], v[46:47], v[48:49]
    v_fma_f64       v[46:47], -v[42:43], v[42:43], v[68:69]
    s_setreg_b32    hwreg(mode, 2, 2), s66
    v_fma_f64       v[42:43], v[46:47], v[48:49], v[42:43]
    v_cmpx_class_f64 v[68:69], s[68:69]
    v_mov_b32       v68, v42
    v_mov_b32       v69, v43
    s_mov_b64       exec, 3
    s_setpc_b64     s[60:61]

fsqrt_r_sub1:
    s_setreg_b32    hwreg(mode, 2, 2), s67
    v_rsq_f64       v[28:29], v[70:71]

    # Improve initial approximation (can be skipped)
    #v_mul_f64       v[42:43], v[28:29], v[70:71]
    #v_mul_f64       v[48:49], v[28:29], -0.5
    #v_fma_f64       v[48:49], v[48:49], v[42:43], 0.5
    #v_fma_f64       v[28:29], v[28:29], v[48:49], v[28:29]

    v_mul_f64       v[42:43], v[28:29], v[70:71]
    v_mov_b32       v48, v28
    v_sub_nc_u32    v49, v29, v84
    v_mov_b32       v46, v28
    v_xor_b32       v47, v49, v82
    v_fma_f64       v[46:47], v[46:47], v[42:43], 0.5
    v_fma_f64       v[42:43], v[42:43], v[46:47], v[42:43]
    v_fma_f64       v[48:49], v[48:49], v[46:47], v[48:49]
    v_fma_f64       v[46:47], -v[42:43], v[42:43], v[70:71]
    s_setreg_b32    hwreg(mode, 2, 2), s66
    v_fma_f64       v[42:43], v[46:47], v[48:49], v[42:43]
    v_cmpx_class_f64 v[70:71], s[68:69]
    v_mov_b32       v70, v42
    v_mov_b32       v71, v43
    s_mov_b64       exec, 3
    s_setpc_b64     s[60:61]

fsqrt_r_sub2:
    s_setreg_b32    hwreg(mode, 2, 2), s67
    v_rsq_f64       v[28:29], v[72:73]

    # Improve initial approximation (can be skipped)
    #v_mul_f64       v[42:43], v[28:29], v[72:73]
    #v_mul_f64       v[48:49], v[28:29], -0.5
    #v_fma_f64       v[48:49], v[48:49], v[42:43], 0.5
    #v_fma_f64       v[28:29], v[28:29], v[48:49], v[28:29]

    v_mul_f64       v[42:43], v[28:29], v[72:73]
    v_mov_b32       v48, v28
    v_sub_nc_u32    v49, v29, v84
    v_mov_b32       v46, v28
    v_xor_b32       v47, v49, v82
    v_fma_f64       v[46:47], v[46:47], v[42:43], 0.5
    v_fma_f64       v[42:43], v[42:43], v[46:47], v[42:43]
    v_fma_f64       v[48:49], v[48:49], v[46:47], v[48:49]
    v_fma_f64       v[46:47], -v[42:43], v[42:43], v[72:73]
    s_setreg_b32    hwreg(mode, 2, 2), s66
    v_fma_f64       v[42:43], v[46:47], v[48:49], v[42:43]
    v_cmpx_class_f64 v[72:73], s[68:69]
    v_mov_b32       v72, v42
    v_mov_b32       v73, v43
    s_mov_b64       exec, 3
    s_setpc_b64     s[60:61]

fsqrt_r_sub3:
    s_setreg_b32    hwreg(mode, 2, 2), s67
    v_rsq_f64       v[28:29], v[74:75]

    # Improve initial approximation (can be skipped)
    #v_mul_f64       v[42:43], v[28:29], v[74:75]
    #v_mul_f64       v[48:49], v[28:29], -0.5
    #v_fma_f64       v[48:49], v[48:49], v[42:43], 0.5
    #v_fma_f64       v[28:29], v[28:29], v[48:49], v[28:29]

    v_mul_f64       v[42:43], v[28:29], v[74:75]
    v_mov_b32       v48, v28
    v_sub_nc_u32    v49, v29, v84
    v_mov_b32       v46, v28
    v_xor_b32       v47, v49, v82
    v_fma_f64       v[46:47], v[46:47], v[42:43], 0.5
    v_fma_f64       v[42:43], v[42:43], v[46:47], v[42:43]
    v_fma_f64       v[48:49], v[48:49], v[46:47], v[48:49]
    v_fma_f64       v[46:47], -v[42:43], v[42:43], v[74:75]
    s_setreg_b32    hwreg(mode, 2, 2), s66
    v_fma_f64       v[42:43], v[46:47], v[48:49], v[42:43]
    v_cmpx_class_f64 v[74:75], s[68:69]
    v_mov_b32       v74, v42
    v_mov_b32       v75, v43
    s_mov_b64       exec, 3
    s_setpc_b64     s[60:61]

fdiv_m_sub0:
    v_or_b32        v28, v28, v78
    v_and_or_b32    v29, v29, v77, v79
    s_setreg_b32    hwreg(mode, 2, 2), s67
    v_rcp_f64       v[48:49], v[28:29]
    v_fma_f64       v[80:81], -v[28:29], v[48:49], 1.0
    v_fma_f64       v[48:49], v[48:49], v[80:81], v[48:49]
    v_mul_f64       v[80:81], v[68:69], v[48:49]
    v_fma_f64       v[42:43], -v[28:29], v[80:81], v[68:69]
    s_setreg_b32    hwreg(mode, 2, 2), s66
    v_fma_f64  v[42:43], v[42:43], v[48:49], v[80:81]
    v_div_fixup_f64 v[80:81], v[42:43], v[28:29], v[68:69]
    v_cmpx_eq_f64   v[68:69], v[28:29]
    v_mov_b32 v80, 0
    v_mov_b32 v81, v83
    s_mov_b64       exec, 3
    v_mov_b32       v68, v80
    v_mov_b32       v69, v81
    s_setpc_b64     s[60:61]

fdiv_m_sub1:
    v_or_b32        v28, v28, v78
    v_and_or_b32    v29, v29, v77, v79
    s_setreg_b32    hwreg(mode, 2, 2), s67
    v_rcp_f64       v[48:49], v[28:29]
    v_fma_f64       v[80:81], -v[28:29], v[48:49], 1.0
    v_fma_f64       v[48:49], v[48:49], v[80:81], v[48:49]
    v_mul_f64       v[80:81], v[70:71], v[48:49]
    v_fma_f64       v[42:43], -v[28:29], v[80:81], v[70:71]
    s_setreg_b32    hwreg(mode, 2, 2), s66
    v_fma_f64  v[42:43], v[42:43], v[48:49], v[80:81]
    v_div_fixup_f64 v[80:81], v[42:43], v[28:29], v[70:71]
    v_cmpx_eq_f64   v[70:71], v[28:29]
    v_mov_b32 v80, 0
    v_mov_b32 v81, v83
    s_mov_b64       exec, 3
    v_mov_b32       v70, v80
    v_mov_b32       v71, v81
    s_setpc_b64     s[60:61]

fdiv_m_sub2:
    v_or_b32        v28, v28, v78
    v_and_or_b32    v29, v29, v77, v79
    s_setreg_b32    hwreg(mode, 2, 2), s67
    v_rcp_f64       v[48:49], v[28:29]
    v_fma_f64       v[80:81], -v[28:29], v[48:49], 1.0
    v_fma_f64       v[48:49], v[48:49], v[80:81], v[48:49]
    v_mul_f64       v[80:81], v[72:73], v[48:49]
    v_fma_f64       v[42:43], -v[28:29], v[80:81], v[72:73]
    s_setreg_b32    hwreg(mode, 2, 2), s66
    v_fma_f64  v[42:43], v[42:43], v[48:49], v[80:81]
    v_div_fixup_f64 v[80:81], v[42:43], v[28:29], v[72:73]
    v_cmpx_eq_f64   v[72:73], v[28:29]
    v_mov_b32 v80, 0
    v_mov_b32 v81, v83
    s_mov_b64       exec, 3
    v_mov_b32       v72, v80
    v_mov_b32       v73, v81
    s_setpc_b64     s[60:61]

fdiv_m_sub3:
    v_or_b32        v28, v28, v78
    v_and_or_b32    v29, v29, v77, v79
    s_setreg_b32    hwreg(mode, 2, 2), s67
    v_rcp_f64       v[48:49], v[28:29]
    v_fma_f64       v[80:81], -v[28:29], v[48:49], 1.0
    v_fma_f64       v[48:49], v[48:49], v[80:81], v[48:49]
    v_mul_f64       v[80:81], v[74:75], v[48:49]
    v_fma_f64       v[42:43], -v[28:29], v[80:81], v[74:75]
    s_setreg_b32    hwreg(mode, 2, 2), s66
    v_fma_f64  v[42:43], v[42:43], v[48:49], v[80:81]
    v_div_fixup_f64 v[80:81], v[42:43], v[28:29], v[74:75]
    v_cmpx_eq_f64   v[74:75], v[28:29]
    v_mov_b32 v80, 0
    v_mov_b32 v81, v83
    s_mov_b64       exec, 3
    v_mov_b32       v74, v80
    v_mov_b32       v75, v81
    s_setpc_b64     s[60:61]

ismulh_r_sub:
    s_mov_b64       exec, 1
    v_mov_b32       v45, s14
    v_mul_hi_u32    v40, s38, v45
    v_mov_b32       v47, s15
    v_mad_u64_u32   v[42:43], s32, s38, v47, v[40:41]
    v_mov_b32       v40, v42
    v_mad_u64_u32   v[45:46], s32, s39, v45, v[40:41]
    v_mad_u64_u32   v[42:43], s32, s39, v47, v[43:44]
    v_add_co_u32    v42, vcc_lo, v42, v46
    v_add_co_ci_u32 v43, vcc_lo, 0, v43, vcc_lo
    v_readlane_b32  s32, v42, 0
    v_readlane_b32  s33, v43, 0
    s_cmp_lt_i32    s15, 0
    s_cselect_b64   s[34:35], s[38:39], 0
    s_sub_u32       s32, s32, s34
    s_subb_u32      s33, s33, s35
    s_cmp_lt_i32    s39, 0
    s_cselect_b64   s[34:35], s[14:15], 0
    s_sub_u32       s14, s32, s34
    s_subb_u32      s15, s33, s35
    s_mov_b64       exec, 3
    s_setpc_b64     s[60:61]

imulh_r_sub:
    s_mov_b64       exec, 1
    v_mov_b32       v45, s38
    v_mul_hi_u32    v40, s14, v45
    v_mov_b32       v47, s39
    v_mad_u64_u32   v[42:43], s32, s14, v47, v[40:41]
    v_mov_b32       v40, v42
    v_mad_u64_u32   v[45:46], s32, s15, v45, v[40:41]
    v_mad_u64_u32   v[42:43], s32, s15, v47, v[43:44]
    v_add_co_u32    v42, vcc_lo, v42, v46
    v_add_co_ci_u32 v43, vcc_lo, 0, v43, vcc_lo
    v_readlane_b32  s14, v42, 0
    v_readlane_b32  s15, v43, 0
    s_mov_b64       exec, 3
    s_setpc_b64     s[60:61]
