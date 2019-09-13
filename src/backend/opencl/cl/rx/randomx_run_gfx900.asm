/*
Copyright (c) 2019 SChernykh

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

.amdcl2
.gpu GFX900
.64bit
.arch_minor 0
.arch_stepping 0
.driver_version 223600
.kernel randomx_run
	.config
		.dims x
		.cws 64, 1, 1
		.sgprsnum 96
		# 6 waves per SIMD: 37-40 VGPRs
		# 5 waves per SIMD: 41-48 VGPRs
		# 4 waves per SIMD: 49-64 VGPRs
		# 3 waves per SIMD: 65-84 VGPRs
		# 2 waves per SIMD: 85-128 VGPRs
		# 1 wave  per SIMD: 129-256 VGPRs
		.vgprsnum 128
		.localsize 256
		.floatmode 0xc0
		.pgmrsrc1 0x00ac035f
		.pgmrsrc2 0x00000090
		.dx10clamp
		.ieeemode
		.useargs
		.priority 0
		.arg _.global_offset_0, "size_t", long
		.arg _.global_offset_1, "size_t", long
		.arg _.global_offset_2, "size_t", long
		.arg _.printf_buffer, "size_t", void*, global, , rdonly
		.arg _.vqueue_pointer, "size_t", long
		.arg _.aqlwrap_pointer, "size_t", long
		.arg dataset, "uchar*", uchar*, global, const, rdonly
		.arg scratchpad, "uchar*", uchar*, global, 
		.arg registers, "ulong*", ulong*, global, 
		.arg rounding_modes, "uint*", uint*, global,
		.arg programs, "uint*", uint*, global, 
		.arg batch_size, "uint", uint
		.arg rx_parameters, "uint", uint
	.text
		s_mov_b32       m0, 0x10000
		s_dcache_wb
		s_waitcnt       vmcnt(0) & lgkmcnt(0)
		s_icache_inv
		s_branch begin

		# pgmrsrc2 = 0x00000090, bits 1:5 = 8, so first 8 SGPRs (s0-s7) contain user data
		# s8 contains group id
		# v0 contains local id
begin:
		v_lshl_add_u32  v1, s8, 6, v0
		s_load_dwordx2  s[0:1], s[4:5], 0x0
		s_load_dwordx2  s[2:3], s[4:5], 0x40
		s_load_dwordx2  s[64:65], s[4:5], 0x48
		s_waitcnt       lgkmcnt(0)

		# load rounding mode
		s_lshl_b32      s16, s8, 2
		s_add_u32       s64, s64, s16
		s_addc_u32      s65, s65, 0
		v_mov_b32       v8, 0
		global_load_dword v8, v8, s[64:65]
		s_waitcnt       vmcnt(0)
		v_readlane_b32  s66, v8, 0
		s_setreg_b32    hwreg(mode, 2, 2), s66
		s_mov_b32       s67, 0

		# used in FSQRT_R to check for "positive normal value" (v_cmpx_class_f64)
		s_mov_b32       s68, 256
		s_mov_b32       s69, 0

		v_add_u32       v1, s0, v1
		v_lshrrev_b32   v2, 6, v1
		v_lshlrev_b32   v3, 5, v2
		v_and_b32       v1, 63, v1
		v_mov_b32       v4, 0
		v_lshlrev_b64   v[3:4], 3, v[3:4]
		v_lshlrev_b32   v5, 4, v1
		v_add_co_u32    v3, vcc, s2, v3
		v_mov_b32       v6, s3
		v_addc_co_u32   v4, vcc, v6, v4, vcc
		v_lshlrev_b32   v41, 2, v1
		v_add_co_u32    v6, vcc, v3, v41
		v_addc_co_u32   v7, vcc, v4, 0, vcc
		global_load_dword v6, v[6:7], off
		v_mov_b32       v0, 0
		s_waitcnt       vmcnt(0)
		ds_write_b32    v41, v6
		s_waitcnt       lgkmcnt(0)
		s_mov_b64       s[0:1], exec
		v_cmpx_le_u32   s[2:3], v1, 7
		s_cbranch_execz program_end

		# rx_parameters
		s_load_dword    s20, s[4:5], 0x5c
		s_waitcnt       lgkmcnt(0)

		# Scratchpad L1 size
		s_bfe_u32       s21, s20, 0x050000
		s_lshl_b32      s21, 1, s21

		# Scratchpad L2 size
		s_bfe_u32       s22, s20, 0x050005
		s_lshl_b32      s22, 1, s22

		# Scratchpad L3 size
		s_bfe_u32       s23, s20, 0x05000A
		s_lshl_b32      s23, 1, s23

		# program iterations
		s_bfe_u32       s24, s20, 0x04000F
		s_lshl_b32      s24, 1, s24

		# Base address for scratchpads
		s_add_u32       s2, s23, 64
		v_mul_hi_u32    v20, v2, s2
		v_mul_lo_u32    v2, v2, s2

		# v41, v44 = 0
		v_mov_b32       v41, 0
		v_mov_b32       v44, 0

		ds_read_b32     v6, v0 offset:152
		v_cmp_lt_u32    s[2:3], v1, 4
		ds_read2_b64    v[34:37], v0 offset0:18 offset1:16
		ds_read_b64     v[11:12], v0 offset:136
		s_movk_i32      s9, 0x0
		s_mov_b64       s[6:7], exec
		s_andn2_b64     exec, s[6:7], s[2:3]
		ds_read_b64     v[13:14], v0 offset:160
		s_andn2_b64     exec, s[6:7], exec
		v_mov_b32       v13, 0
		v_mov_b32       v14, 0
		s_mov_b64       exec, s[6:7]

		# compiled program size
		s_mov_b64       s[6:7], s[8:9]
		s_mulk_i32      s6, 10048

		v_add3_u32      v5, v0, v5, 64
		s_mov_b64       s[8:9], exec
		s_andn2_b64     exec, s[8:9], s[2:3]
		ds_read_b64     v[15:16], v0 offset:168
		s_andn2_b64     exec, s[8:9], exec
		v_mov_b32       v15, 0
		v_mov_b32       v16, 0
		s_mov_b64       exec, s[8:9]
		s_load_dwordx4  s[8:11], s[4:5], 0x30

		# batch_size
		s_load_dword    s16, s[4:5], 0x58

		s_load_dwordx2  s[4:5], s[4:5], 0x50
		v_lshlrev_b32   v1, 3, v1
		v_add_u32       v17, v0, v1
		s_waitcnt       lgkmcnt(0)
		v_add_co_u32    v2, vcc, s10, v2
		v_mov_b32       v18, s11
		v_addc_co_u32   v18, vcc, v18, v20, vcc
		v_mov_b32       v19, 0xffffff
		v_add_co_u32    v6, vcc, s8, v6
		v_mov_b32       v20, s9
		v_addc_co_u32   v20, vcc, v20, 0, vcc
		ds_read_b64     v[21:22], v17
		s_add_u32       s4, s4, s6
		s_addc_u32      s5, s5, s7
		v_cndmask_b32   v19, v19, -1, s[2:3]
		v_lshl_add_u32  v8, v35, 3, v0
		v_lshl_add_u32  v7, v34, 3, v0
		v_lshl_add_u32  v12, v12, 3, v0
		v_lshl_add_u32  v0, v11, 3, v0
		v_mov_b32       v10, v36
		v_mov_b32       v23, v37

		# loop counter
		s_sub_u32       s2, s24, 1

		# batch_size
		s_mov_b32       s3, s16

		# Scratchpad masks for scratchpads
		v_sub_u32       v38, s21, 8
		v_sub_u32       v39, s22, 8
		v_sub_u32       v50, s23, 8

		# mask for FSCAL_R
		v_mov_b32       v51, 0x80F00000

		# load scratchpad base address
		v_readlane_b32	s0, v2, 0
		v_readlane_b32	s1, v18, 0

		# save current executiom mask
		s_mov_b64       s[36:37], exec

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
		s_mov_b64       exec, s[36:37]

		# sign mask (used in FSQRT_R)
		v_mov_b32       v82, 0x80000000

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

main_loop:
		# const uint2 spMix = as_uint2(R[readReg0] ^ R[readReg1]);
		ds_read_b64     v[24:25], v0
		ds_read_b64     v[26:27], v12
		s_waitcnt       lgkmcnt(0)
		v_xor_b32       v25, v27, v25
		v_xor_b32       v24, v26, v24

		# spAddr1 ^= spMix.y;
		# spAddr0 ^= spMix.x;
		v_xor_b32       v10, v25, v10
		v_xor_b32       v23, v24, v23

		# spAddr1 &= ScratchpadL3Mask64;
		# spAddr0 &= ScratchpadL3Mask64;
		v_and_b32       v10, s86, v10
		v_and_b32       v23, s86, v23

		# Offset for scratchpads
		# offset1 = spAddr1 + sub * 8
		# offset0 = spAddr0 + sub * 8
		v_add_u32       v10, v10, v1
		v_add_u32       v23, v23, v1

		# __global ulong* p1 = (__global ulong*)(scratchpad + offset1);
		# __global ulong* p0 = (__global ulong*)(scratchpad + offset0);
		v_add_co_u32    v26, vcc, v2, v10
		v_addc_co_u32   v27, vcc, v18, 0, vcc
		v_add_co_u32    v23, vcc, v2, v23
		v_addc_co_u32   v24, vcc, v18, 0, vcc

		# load from spAddr1
		global_load_dwordx2 v[28:29], v[26:27], off

		# load from spAddr0
		global_load_dwordx2 v[30:31], v[23:24], off
		s_waitcnt       vmcnt(1)

		v_cvt_f64_i32   v[32:33], v28
		v_cvt_f64_i32   v[28:29], v29
		s_waitcnt       vmcnt(0)

		# R[sub] ^= *p0;
		v_xor_b32       v34, v21, v30
		v_xor_b32       v35, v22, v31

		v_add_co_u32    v22, vcc, v6, v36
		v_addc_co_u32   v25, vcc, v20, 0, vcc
		v_add_co_u32    v21, vcc, v22, v1
		v_addc_co_u32   v22, vcc, v25, 0, vcc
		global_load_dwordx2 v[21:22], v[21:22], off
		v_or_b32        v30, v32, v13
		v_and_or_b32    v31, v33, v19, v14
		v_or_b32        v28, v28, v15
		v_and_or_b32    v29, v29, v19, v16
		ds_write2_b64   v5, v[30:31], v[28:29] offset1:1
		s_waitcnt       lgkmcnt(0)

		# Program 0

		# load group F,E registers
		# Read low 8 bytes into lane 0 and high 8 bytes into lane 1
		s_mov_b64       exec, 3
		ds_read2_b64    v[60:63], v41 offset0:8 offset1:10
		ds_read2_b64    v[64:67], v41 offset0:12 offset1:14
		ds_read2_b64    v[68:71], v41 offset0:16 offset1:18
		ds_read2_b64    v[72:75], v41 offset0:20 offset1:22

		# load VM integer registers
		v_readlane_b32	s16, v34, 0
		v_readlane_b32	s17, v35, 0
		v_readlane_b32	s18, v34, 1
		v_readlane_b32	s19, v35, 1
		v_readlane_b32	s20, v34, 2
		v_readlane_b32	s21, v35, 2
		v_readlane_b32	s22, v34, 3
		v_readlane_b32	s23, v35, 3
		v_readlane_b32	s24, v34, 4
		v_readlane_b32	s25, v35, 4
		v_readlane_b32	s26, v34, 5
		v_readlane_b32	s27, v35, 5
		v_readlane_b32	s28, v34, 6
		v_readlane_b32	s29, v35, 6
		v_readlane_b32	s30, v34, 7
		v_readlane_b32	s31, v35, 7

		s_waitcnt       lgkmcnt(0)

		# call JIT code
		s_swappc_b64    s[12:13], s[4:5]

		# Write out group F,E registers
		# Write low 8 bytes from lane 0 and high 8 bytes from lane 1
		ds_write2_b64   v41, v[60:61], v[62:63] offset0:8 offset1:10
		ds_write2_b64   v41, v[64:65], v[66:67] offset0:12 offset1:14
		ds_write2_b64   v41, v[68:69], v[70:71] offset0:16 offset1:18
		ds_write2_b64   v41, v[72:73], v[74:75] offset0:20 offset1:22

		# store VM integer registers
		v_writelane_b32 v28, s16, 0
		v_writelane_b32 v29, s17, 0
		v_writelane_b32 v28, s18, 1
		v_writelane_b32 v29, s19, 1
		v_writelane_b32 v28, s20, 2
		v_writelane_b32 v29, s21, 2
		v_writelane_b32 v28, s22, 3
		v_writelane_b32 v29, s23, 3
		v_writelane_b32 v28, s24, 4
		v_writelane_b32 v29, s25, 4
		v_writelane_b32 v28, s26, 5
		v_writelane_b32 v29, s27, 5
		v_writelane_b32 v28, s28, 6
		v_writelane_b32 v29, s29, 6
		v_writelane_b32 v28, s30, 7
		v_writelane_b32 v29, s31, 7

		# Restore execution mask
		s_mov_b64       exec, s[36:37]

		# Write out VM integer registers
		ds_write_b64    v17, v[28:29]

		s_waitcnt       lgkmcnt(0)
		v_xor_b32       v21, v28, v21
		v_xor_b32       v22, v29, v22
		ds_read_b32     v28, v7
		ds_read_b32     v29, v8
		ds_write_b64    v17, v[21:22]
		s_waitcnt       lgkmcnt(1)
		ds_read2_b64    v[30:33], v17 offset0:8 offset1:16
		v_xor_b32       v10, v28, v37
		s_waitcnt       lgkmcnt(0)
		v_xor_b32       v30, v32, v30
		v_xor_b32       v31, v33, v31
		v_xor_b32       v10, v10, v29
		global_store_dwordx2 v[26:27], v[21:22], off
		v_and_b32       v10, 0x7fffffc0, v10
		global_store_dwordx2 v[23:24], v[30:31], off
		s_cmp_eq_u32    s2, 0
		s_cbranch_scc1  main_loop_end
		s_sub_i32       s2, s2, 1
		v_mov_b32       v37, v36
		v_mov_b32       v23, 0
		v_mov_b32       v36, v10
		v_mov_b32       v10, 0
		s_branch        main_loop
main_loop_end:

		v_add_co_u32    v0, vcc, v3, v1
		v_addc_co_u32   v1, vcc, v4, 0, vcc
		global_store_dwordx2 v[0:1], v[21:22], off
		global_store_dwordx2 v[0:1], v[30:31], off inst_offset:64
		global_store_dwordx2 v[0:1], v[32:33], off inst_offset:128

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
		v_sub_u32       v49, v29, v84
		v_mov_b32       v46, v28
		v_xor_b32       v47, v49, v82
		v_fma_f64       v[46:47], v[46:47], v[42:43], 0.5
		v_fma_f64       v[42:43], v[42:43], v[46:47], v[42:43]
		v_fma_f64       v[48:49], v[48:49], v[46:47], v[48:49]
		v_fma_f64       v[46:47], -v[42:43], v[42:43], v[68:69]
		s_setreg_b32    hwreg(mode, 2, 2), s66
		v_fma_f64       v[42:43], v[46:47], v[48:49], v[42:43]
		v_cmpx_class_f64 s[14:15], v[68:69], s[68:69]
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
		v_sub_u32       v49, v29, v84
		v_mov_b32       v46, v28
		v_xor_b32       v47, v49, v82
		v_fma_f64       v[46:47], v[46:47], v[42:43], 0.5
		v_fma_f64       v[42:43], v[42:43], v[46:47], v[42:43]
		v_fma_f64       v[48:49], v[48:49], v[46:47], v[48:49]
		v_fma_f64       v[46:47], -v[42:43], v[42:43], v[70:71]
		s_setreg_b32    hwreg(mode, 2, 2), s66
		v_fma_f64       v[42:43], v[46:47], v[48:49], v[42:43]
		v_cmpx_class_f64 s[14:15], v[70:71], s[68:69]
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
		v_sub_u32       v49, v29, v84
		v_mov_b32       v46, v28
		v_xor_b32       v47, v49, v82
		v_fma_f64       v[46:47], v[46:47], v[42:43], 0.5
		v_fma_f64       v[42:43], v[42:43], v[46:47], v[42:43]
		v_fma_f64       v[48:49], v[48:49], v[46:47], v[48:49]
		v_fma_f64       v[46:47], -v[42:43], v[42:43], v[72:73]
		s_setreg_b32    hwreg(mode, 2, 2), s66
		v_fma_f64       v[42:43], v[46:47], v[48:49], v[42:43]
		v_cmpx_class_f64 s[14:15], v[72:73], s[68:69]
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
		v_sub_u32       v49, v29, v84
		v_mov_b32       v46, v28
		v_xor_b32       v47, v49, v82
		v_fma_f64       v[46:47], v[46:47], v[42:43], 0.5
		v_fma_f64       v[42:43], v[42:43], v[46:47], v[42:43]
		v_fma_f64       v[48:49], v[48:49], v[46:47], v[48:49]
		v_fma_f64       v[46:47], -v[42:43], v[42:43], v[74:75]
		s_setreg_b32    hwreg(mode, 2, 2), s66
		v_fma_f64       v[42:43], v[46:47], v[48:49], v[42:43]
		v_cmpx_class_f64 s[14:15], v[74:75], s[68:69]
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
		v_cmpx_eq_f64   s[14:15], v[68:69], v[28:29]
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
		v_cmpx_eq_f64   s[14:15], v[70:71], v[28:29]
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
		v_cmpx_eq_f64   s[14:15], v[72:73], v[28:29]
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
		v_cmpx_eq_f64   s[14:15], v[74:75], v[28:29]
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
		v_mad_u64_u32   v[42:43], s[32:33], s38, v47, v[40:41]
		v_mov_b32       v40, v42
		v_mad_u64_u32   v[45:46], s[32:33], s39, v45, v[40:41]
		v_mad_u64_u32   v[42:43], s[32:33], s39, v47, v[43:44]
		v_add_co_u32    v42, vcc, v42, v46
		v_addc_co_u32   v43, vcc, 0, v43, vcc
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
		v_mad_u64_u32   v[42:43], s[32:33], s14, v47, v[40:41]
		v_mov_b32       v40, v42
		v_mad_u64_u32   v[45:46], s[32:33], s15, v45, v[40:41]
		v_mad_u64_u32   v[42:43], s[32:33], s15, v47, v[43:44]
		v_add_co_u32    v42, vcc, v42, v46
		v_addc_co_u32   v43, vcc, 0, v43, vcc
		v_readlane_b32  s14, v42, 0
		v_readlane_b32  s15, v43, 0
		s_mov_b64       exec, 3
		s_setpc_b64     s[60:61]
