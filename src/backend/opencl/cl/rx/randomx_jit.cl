/*
Copyright (c) 2019 SChernykh
Portions Copyright (c) 2018-2019 tevador

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

#define INITIAL_HASH_SIZE 64
#define INTERMEDIATE_PROGRAM_SIZE (RANDOMX_PROGRAM_SIZE * 16)
#define COMPILED_PROGRAM_SIZE 10048
#define NUM_VGPR_REGISTERS 128

#define mantissaSize 52
#define exponentSize 11
#define mantissaMask ((1UL << mantissaSize) - 1)
#define exponentMask ((1UL << exponentSize) - 1)
#define exponentBias 1023

#define dynamicExponentBits 4
#define staticExponentBits 4
#define constExponentBits 0x300
#define dynamicMantissaMask ((1UL << (mantissaSize + dynamicExponentBits)) - 1)

#define ScratchpadL1Mask_reg 38
#define ScratchpadL2Mask_reg 39
#define ScratchpadL3Mask_reg 50

#define ScratchpadL3Mask (RANDOMX_SCRATCHPAD_L3 - 8)

#define RANDOMX_JUMP_BITS          8
#define RANDOMX_JUMP_OFFSET        8

#if GCN_VERSION >= 15

#define S_SETPC_B64_S12_13 0xbe80200cu
#define V_AND_B32_CALC_ADDRESS 0x3638000eu
#define GLOBAL_LOAD_DWORDX2_SCRATCHPAD_LOAD 0xdc348000u
#define S_WAITCNT_SCRATCHPAD_LOAD2 0xbf8c3f70u
#define V_READLANE_B32_SCRATCHPAD_LOAD2 0xd7600000u
#define S_MUL_HI_U32_IMUL_R 0x9a8f1010u
#define S_MUL_I32_IMUL 0x93000000u
#define S_MUL_HI_U32_IMUL_R_2 0x9a8fff10u
#define S_MUL_HI_U32_IMUL_M 0x9aa10e10u
#define S_MOV_B32_IMUL_RCP 0xbea003ffu
#define S_MUL_HI_U32_IMUL_RCP 0x9a8f2010u
#define S_XOR_B32_64 0x89000000u
#define S_MOV_B32_XOR_R 0xbebe03ffu
#define S_LSHR 0x90000000u
#define S_LSHL 0x8f000000u
#define S_OR 0x88000000u
#define S_AND 0x87000000u
#define S_BFE 0x94000000u
#define DS_SWIZZLE_B32_FSWAP_R 0xd8d48001u
#define V_ADD_F64 0xd564003cu
#define V_AND_B32 0x36000000u
#define GLOBAL_LOAD_DWORD_SCRATCHPAD_LOAD_FP 0xdc308000u
#define V_XOR_B32 0x3a000000u
#define V_MUL_F64 0xd5650044u

#else

#define S_SETPC_B64_S12_13 0xbe801d0cu
#define V_AND_B32_CALC_ADDRESS 0x2638000eu
#define GLOBAL_LOAD_DWORDX2_SCRATCHPAD_LOAD 0xdc548000u
#define S_WAITCNT_SCRATCHPAD_LOAD2 0xbf8c0f70u
#define V_READLANE_B32_SCRATCHPAD_LOAD2 0xd2890000u
#define S_MUL_HI_U32_IMUL_R 0x960f1010u
#define S_MUL_I32_IMUL 0x92000000u
#define S_MUL_HI_U32_IMUL_R_2 0x960fff10u
#define S_MUL_HI_U32_IMUL_M 0x96210e10u
#define S_MOV_B32_IMUL_RCP 0xbea000ffu
#define S_MUL_HI_U32_IMUL_RCP 0x960f2010u
#define S_XOR_B32_64 0x88000000u
#define S_MOV_B32_XOR_R 0xbebe00ffu
#define S_LSHR 0x8f000000u
#define S_LSHL 0x8e000000u
#define S_OR 0x87000000u
#define S_AND 0x86000000u
#define S_BFE 0x93000000u
#define DS_SWIZZLE_B32_FSWAP_R 0xd87a8001u
#define V_ADD_F64 0xd280003cu
#define V_AND_B32 0x26000000u
#define GLOBAL_LOAD_DWORD_SCRATCHPAD_LOAD_FP 0xdc508000u
#define V_XOR_B32 0x2a000000u
#define V_MUL_F64 0xd2810044u

#endif

__global uint* jit_scratchpad_calc_address(__global uint* p, uint src, uint imm32, uint mask_reg, uint batch_size)
{
	// s_add_i32 s14, s(16 + src * 2), imm32
	*(p++) = 0x810eff10u | (src << 1);
	*(p++) = imm32;

	// v_and_b32 v28, s14, mask_reg
	*(p++) = V_AND_B32_CALC_ADDRESS | (mask_reg << 9);

	return p;
}

__global uint* jit_scratchpad_calc_fixed_address(__global uint* p, uint imm32, uint batch_size)
{
	// v_mov_b32 v28, imm32
	*(p++) = 0x7e3802ffu;
	*(p++) = imm32;

	return p;
}

__global uint* jit_scratchpad_load(__global uint* p, uint vgpr_index)
{
	// v28 = offset

#if GCN_VERSION >= 14
	// global_load_dwordx2 v[vgpr_index:vgpr_index+1], v28, s[0:1]
	*(p++) = GLOBAL_LOAD_DWORDX2_SCRATCHPAD_LOAD;
	*(p++) = 0x0000001cu | (vgpr_index << 24);
#else
	*(p++) = 0x32543902u;						// v_add_u32 v42, vcc, v2, v28
	*(p++) = 0xd11c6a2bu;						// v_addc_u32 v43, vcc, v3, 0, vcc
	*(p++) = 0x01a90103u;
	*(p++) = 0xdc540000u;						// flat_load_dwordx2 v[vgpr_index:vgpr_index+1], v[42:43]
	*(p++) = 0x0000002au | (vgpr_index << 24);
#endif

	return p;
}

__global uint* jit_scratchpad_load2(__global uint* p, uint vgpr_index, int vmcnt)
{
	// s_waitcnt vmcnt(N)
	if (vmcnt >= 0)
		*(p++) = S_WAITCNT_SCRATCHPAD_LOAD2 | (vmcnt & 15) | ((vmcnt >> 4) << 14);

	// v_readlane_b32 s14, vgpr_index, 0
	*(p++) = V_READLANE_B32_SCRATCHPAD_LOAD2 | 14;
	*(p++) = 0x00010100u | vgpr_index;

	// v_readlane_b32 s15, vgpr_index + 1, 0
	*(p++) = V_READLANE_B32_SCRATCHPAD_LOAD2 | 15;
	*(p++) = 0x00010100u | (vgpr_index + 1);

	return p;
}

__global uint* jit_scratchpad_calc_address_fp(__global uint* p, uint src, uint imm32, uint mask_reg, uint batch_size)
{
	// s_add_i32 s14, s(16 + src * 2), imm32
	*(p++) = 0x810eff10u | (src << 1);
	*(p++) = imm32;

	// v_and_b32 v28, s14, mask_reg
	*(p++) = V_AND_B32 | 0x38000eu | (mask_reg << 9);

#if GCN_VERSION >= 15
	// v_add_nc_u32 v28, v28, v44
	*(p++) = 0x4a38591cu;
#elif GCN_VERSION == 14
	// v_add_u32 v28, v28, v44
	*(p++) = 0x6838591cu;
#else
	// v_add_u32 v28, vcc, v28, v44
	*(p++) = 0x3238591cu;
#endif

	return p;
}

__global uint* jit_scratchpad_load_fp(__global uint* p, uint vgpr_index)
{
	// v28 = offset

#if GCN_VERSION >= 14
	// global_load_dword v(vgpr_index), v28, s[0:1]
	*(p++) = GLOBAL_LOAD_DWORD_SCRATCHPAD_LOAD_FP;
	*(p++) = 0x0000001cu | (vgpr_index << 24);
#else
	*(p++) = 0x32543902u;						// v_add_u32 v42, vcc, v2, v28
	*(p++) = 0xd11c6a2bu;						// v_addc_u32 v43, vcc, v3, 0, vcc
	*(p++) = 0x01a90103u;
	*(p++) = 0xdc500000u;						// flat_load_dword v(vgpr_index), v[42:43]
	*(p++) = 0x0000002au | (vgpr_index << 24);
#endif

	return p;
}

__global uint* jit_scratchpad_load2_fp(__global uint* p, uint vgpr_index, int vmcnt)
{
	// s_waitcnt vmcnt(N)
	if (vmcnt >= 0)
		*(p++) = S_WAITCNT_SCRATCHPAD_LOAD2 | (vmcnt & 15) | ((vmcnt >> 4) << 14);

	// v_cvt_f64_i32 v[28:29], vgpr_index
	*(p++) = 0x7e380900u | vgpr_index;

	return p;
}

__global uint* jit_emit_instruction(__global uint* p, __global uint* last_branch_target, const uint2 inst, int prefetch_vgpr_index, int vmcnt, uint batch_size)
{
	uint opcode = inst.x & 0xFF;
	const uint dst = (inst.x >> 8) & 7;
	const uint src = (inst.x >> 16) & 7;
	const uint mod = inst.x >> 24;

	if (opcode < RANDOMX_FREQ_IADD_RS)
	{
		const uint shift = (mod >> 2) % 4;
		if (shift > 0) // p = 3/4
		{
			// s_lshl_b64 s[14:15], s[(16 + src * 2):(17 + src * 2)], shift
			*(p++) = S_LSHL | 0x8e8010u | (src << 1) | (shift << 8);

			// s_add_u32 s(16 + dst * 2), s(16 + dst * 2), s14
			*(p++) = 0x80100e10u | (dst << 1) | (dst << 17);

			// s_addc_u32 s(17 + dst * 2), s(17 + dst * 2), s15
			*(p++) = 0x82110f11u | (dst << 1) | (dst << 17);
		}
		else // p = 1/4
		{
			// s_add_u32 s(16 + dst * 2), s(16 + dst * 2), s(16 + src * 2)
			*(p++) = 0x80101010u | (dst << 1) | (dst << 17) | (src << 9);

			// s_addc_u32 s(17 + dst * 2), s(17 + dst * 2), s(17 + src * 2)
			*(p++) = 0x82111111u | (dst << 1) | (dst << 17) | (src << 9);
		}

		if (dst == 5) // p = 1/8
		{
			// s_add_u32 s(16 + dst * 2), s(16 + dst * 2), imm32
			*(p++) = 0x8010ff10u | (dst << 1) | (dst << 17);
			*(p++) = inst.y;

			// s_addc_u32 s(17 + dst * 2), s(17 + dst * 2), ((inst.y < 0) ? -1 : 0)
			*(p++) = 0x82110011u | (dst << 1) | (dst << 17) | (((as_int(inst.y) < 0) ? 0xc1 : 0x80) << 8);
		}

		// 12*3/4 + 8*1/4 + 12/8 = 12.5 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_IADD_RS;

	if (opcode < RANDOMX_FREQ_IADD_M)
	{
		if (prefetch_vgpr_index >= 0)
		{
			if (src != dst) // p = 7/8
				p = jit_scratchpad_calc_address(p, src, inst.y, (mod % 4) ? ScratchpadL1Mask_reg : ScratchpadL2Mask_reg, batch_size);
			else // p = 1/8
				p = jit_scratchpad_calc_fixed_address(p, inst.y & ScratchpadL3Mask, batch_size);

			p = jit_scratchpad_load(p, prefetch_vgpr_index ? prefetch_vgpr_index : 28);
		}

		if (prefetch_vgpr_index <= 0)
		{
			p = jit_scratchpad_load2(p, prefetch_vgpr_index ? -prefetch_vgpr_index : 28, prefetch_vgpr_index ? vmcnt : 0);

			// s_add_u32 s(16 + dst * 2), s(16 + dst * 2), s14
			*(p++) = 0x80100e10u | (dst << 1) | (dst << 17);

			// s_addc_u32 s(17 + dst * 2), s(17 + dst * 2), s15
			*(p++) = 0x82110f11u | (dst << 1) | (dst << 17);
		}

		// (12*7/8 + 8*1/8 + 28) + 8 = 47.5 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_IADD_M;

	if (opcode < RANDOMX_FREQ_ISUB_R)
	{
		if (src != dst) // p = 7/8
		{
			// s_sub_u32 s(16 + dst * 2), s(16 + dst * 2), s(16 + src * 2)
			*(p++) = 0x80901010u | (dst << 1) | (dst << 17) | (src << 9);

			// s_subb_u32 s(17 + dst * 2), s(17 + dst * 2), s(17 + src * 2)
			*(p++) = 0x82911111u | (dst << 1) | (dst << 17) | (src << 9);
		}
		else // p = 1/8
		{
			// s_sub_u32 s(16 + dst * 2), s(16 + dst * 2), imm32
			*(p++) = 0x8090ff10u | (dst << 1) | (dst << 17);
			*(p++) = inst.y;

			// s_subb_u32 s(17 + dst * 2), s(17 + dst * 2), ((inst.y < 0) ? -1 : 0)
			*(p++) = 0x82910011u | (dst << 1) | (dst << 17) | (((as_int(inst.y) < 0) ? 0xc1 : 0x80) << 8);
		}

		// 8*7/8 + 12/8 = 8.5 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_ISUB_R;

	if (opcode < RANDOMX_FREQ_ISUB_M)
	{
		if (prefetch_vgpr_index >= 0)
		{
			if (src != dst) // p = 7/8
				p = jit_scratchpad_calc_address(p, src, inst.y, (mod % 4) ? ScratchpadL1Mask_reg : ScratchpadL2Mask_reg, batch_size);
			else // p = 1/8
				p = jit_scratchpad_calc_fixed_address(p, inst.y & ScratchpadL3Mask, batch_size);

			p = jit_scratchpad_load(p, prefetch_vgpr_index ? prefetch_vgpr_index : 28);
		}

		if (prefetch_vgpr_index <= 0)
		{
			p = jit_scratchpad_load2(p, prefetch_vgpr_index ? -prefetch_vgpr_index : 28, prefetch_vgpr_index ? vmcnt : 0);

			// s_sub_u32 s(16 + dst * 2), s(16 + dst * 2), s14
			*(p++) = 0x80900e10u | (dst << 1) | (dst << 17);

			// s_subb_u32 s(17 + dst * 2), s(17 + dst * 2), s15
			*(p++) = 0x82910f11u | (dst << 1) | (dst << 17);
		}

		// (12*7/8 + 8*1/8 + 28) + 8 = 47.5 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_ISUB_M;

	if (opcode < RANDOMX_FREQ_IMUL_R)
	{
		if (src != dst) // p = 7/8
		{
#if GCN_VERSION >= 14
			// s_mul_hi_u32 s15, s(16 + dst * 2), s(16 + src * 2)
			*(p++) = S_MUL_HI_U32_IMUL_R | (dst << 1) | (src << 9);
#else
			// v_mov_b32 v28, s(16 + dst * 2)
			*(p++) = 0x7e380210u | (dst << 1);
			// v_mul_hi_u32 v28, v28, s(16 + src * 2)
			*(p++) = 0xd286001cu;
			*(p++) = 0x0000211cu + (src << 10);
			// v_readlane_b32 s15, v28, 0
			*(p++) = 0xd289000fu;
			*(p++) = 0x0001011cu;
#endif

			// s_mul_i32 s14, s(16 + dst * 2), s(17 + src * 2)
			*(p++) = S_MUL_I32_IMUL | 0x0e1110u | (dst << 1) | (src << 9);

			// s_add_u32 s15, s15, s14
			*(p++) = 0x800f0e0fu;

			// s_mul_i32 s14, s(17 + dst * 2), s(16 + src * 2)
			*(p++) = S_MUL_I32_IMUL | 0x0e1011u | (dst << 1) | (src << 9);

			// s_add_u32 s(17 + dst * 2), s15, s14
			*(p++) = 0x80110e0fu | (dst << 17);

			// s_mul_i32 s(16 + dst * 2), s(16 + dst * 2), s(16 + src * 2)
			*(p++) = S_MUL_I32_IMUL | 0x101010u | (dst << 1) | (dst << 17) | (src << 9);
		}
		else // p = 1/8
		{
#if GCN_VERSION >= 14
			// s_mul_hi_u32 s15, s(16 + dst * 2), imm32
			*(p++) = S_MUL_HI_U32_IMUL_R_2 | (dst << 1);
			*(p++) = inst.y;
#else
			// v_mov_b32 v28, imm32
			*(p++) = 0x7e3802ffu;
			*(p++) = inst.y;
			// v_mul_hi_u32 v28, v28, s(16 + dst * 2)
			*(p++) = 0xd286001cu;
			*(p++) = 0x0000211cu + (dst << 10);
			// v_readlane_b32 s15, v28, 0
			*(p++) = 0xd289000fu;
			*(p++) = 0x0001011cu;
#endif

			if (as_int(inst.y) < 0) // p = 1/2
			{
				// s_sub_u32 s15, s15, s(16 + dst * 2)
				*(p++) = 0x808f100fu | (dst << 9);
			}

			// s_mul_i32 s14, s(17 + dst * 2), imm32
			*(p++) = S_MUL_I32_IMUL | 0x0eff11u | (dst << 1);
			*(p++) = inst.y;

			// s_add_u32 s(17 + dst * 2), s15, s14
			*(p++) = 0x80110e0fu | (dst << 17);

			// s_mul_i32 s(16 + dst * 2), s(16 + dst * 2), imm32
			*(p++) = S_MUL_I32_IMUL | 0x10ff10u | (dst << 1) | (dst << 17);
			*(p++) = inst.y;
		}

		// 24*7/8 + 28*1/8 + 4*1/16 = 24.75 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_IMUL_R;

	if (opcode < RANDOMX_FREQ_IMUL_M)
	{
		if (prefetch_vgpr_index >= 0)
		{
			if (src != dst) // p = 7/8
				p = jit_scratchpad_calc_address(p, src, inst.y, (mod % 4) ? ScratchpadL1Mask_reg : ScratchpadL2Mask_reg, batch_size);
			else // p = 1/8
				p = jit_scratchpad_calc_fixed_address(p, inst.y & ScratchpadL3Mask, batch_size);

			p = jit_scratchpad_load(p, prefetch_vgpr_index ? prefetch_vgpr_index : 28);
		}

		if (prefetch_vgpr_index <= 0)
		{
			p = jit_scratchpad_load2(p, prefetch_vgpr_index ? -prefetch_vgpr_index : 28, prefetch_vgpr_index ? vmcnt : 0);

#if GCN_VERSION >= 14
			// s_mul_hi_u32 s33, s(16 + dst * 2), s14
			*(p++) = S_MUL_HI_U32_IMUL_M | (dst << 1);
#else
			// v_mov_b32 v28, s(16 + dst * 2)
			*(p++) = 0x7e380210u | (dst << 1);
			// v_mul_hi_u32 v28, v28, s14
			*(p++) = 0xd286001cu;
			*(p++) = 0x00001d1cu;
			// v_readlane_b32 s33, v28, 0
			*(p++) = 0xd2890021u;
			*(p++) = 0x0001011cu;
#endif

			// s_mul_i32 s32, s(16 + dst * 2), s15
			*(p++) = S_MUL_I32_IMUL | 0x200f10u | (dst << 1);

			// s_add_u32 s33, s33, s32
			*(p++) = 0x80212021u;

			// s_mul_i32 s32, s(17 + dst * 2), s14
			*(p++) = S_MUL_I32_IMUL | 0x200e11u | (dst << 1);

			// s_add_u32 s(17 + dst * 2), s33, s32
			*(p++) = 0x80112021u | (dst << 17);

			// s_mul_i32 s(16 + dst * 2), s(16 + dst * 2), s14
			*(p++) = S_MUL_I32_IMUL | 0x100e10u | (dst << 1) | (dst << 17);
		}

		// (12*7/8 + 8*1/8 + 28) + 24 = 63.5 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_IMUL_M;

	if (opcode < RANDOMX_FREQ_IMULH_R)
	{
#if GCN_VERSION >= 15
		*(p++) = 0xbe8e0410u | (dst << 1);				// s_mov_b64 s[14:15], s[16 + dst * 2:17 + dst * 2]
		*(p++) = 0xbea60410u | (src << 1);				// s_mov_b64 s[38:39], s[16 + src * 2:17 + src * 2]
		*(p++) = 0xbebc213au;							// s_swappc_b64 s[60:61], s[58:59]
		*(p++) = 0xbe90040eu | (dst << 17);				// s_mov_b64 s[16 + dst * 2:17 + dst * 2], s[14:15]
#else
		*(p++) = 0xbe8e0110u | (dst << 1);				// s_mov_b64 s[14:15], s[16 + dst * 2:17 + dst * 2]
		*(p++) = 0xbea60110u | (src << 1);				// s_mov_b64 s[38:39], s[16 + src * 2:17 + src * 2]
		*(p++) = 0xbebc1e3au;							// s_swappc_b64 s[60:61], s[58:59]
		*(p++) = 0xbe90010eu | (dst << 17);				// s_mov_b64 s[16 + dst * 2:17 + dst * 2], s[14:15]
#endif

		// 16 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_IMULH_R;

	if (opcode < RANDOMX_FREQ_IMULH_M)
	{
		if (prefetch_vgpr_index >= 0)
		{
			if (src != dst) // p = 7/8
				p = jit_scratchpad_calc_address(p, src, inst.y, (mod % 4) ? ScratchpadL1Mask_reg : ScratchpadL2Mask_reg, batch_size);
			else // p = 1/8
				p = jit_scratchpad_calc_fixed_address(p, inst.y & ScratchpadL3Mask, batch_size);

			p = jit_scratchpad_load(p, prefetch_vgpr_index ? prefetch_vgpr_index : 28);
		}

		if (prefetch_vgpr_index <= 0)
		{
			p = jit_scratchpad_load2(p, prefetch_vgpr_index ? -prefetch_vgpr_index : 28, prefetch_vgpr_index ? vmcnt : 0);

#if GCN_VERSION >= 15
			*(p++) = 0xbea60410u | (dst << 1);				// s_mov_b64 s[38:39], s[16 + src * 2:17 + src * 2]
			*(p++) = 0xbebc213au;							// s_swappc_b64 s[60:61], s[58:59]
			*(p++) = 0xbe90040eu | (dst << 17);				// s_mov_b64 s[16 + dst * 2:17 + dst * 2], s[14:15]
#else
			*(p++) = 0xbea60110u | (dst << 1);				// s_mov_b64 s[38:39], s[16 + src * 2:17 + src * 2]
			*(p++) = 0xbebc1e3au;							// s_swappc_b64 s[60:61], s[58:59]
			*(p++) = 0xbe90010eu | (dst << 17);				// s_mov_b64 s[16 + dst * 2:17 + dst * 2], s[14:15]
#endif
		}

		// (12*7/8 + 8*1/8 + 28) + 12 = 51.5 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_IMULH_M;

	if (opcode < RANDOMX_FREQ_ISMULH_R)
	{
#if GCN_VERSION >= 15
		*(p++) = 0xbe8e0410u | (dst << 1);				// s_mov_b64 s[14:15], s[16 + dst * 2:17 + dst * 2]
		*(p++) = 0xbea60410u | (src << 1);				// s_mov_b64 s[38:39], s[16 + src * 2:17 + src * 2]
		*(p++) = 0xbebc2138u;							// s_swappc_b64 s[60:61], s[56:57]
		*(p++) = 0xbe90040eu | (dst << 17);				// s_mov_b64 s[16 + dst * 2:17 + dst * 2], s[14:15]
#else
		*(p++) = 0xbe8e0110u | (dst << 1);				// s_mov_b64 s[14:15], s[16 + dst * 2:17 + dst * 2]
		*(p++) = 0xbea60110u | (src << 1);				// s_mov_b64 s[38:39], s[16 + src * 2:17 + src * 2]
		*(p++) = 0xbebc1e38u;							// s_swappc_b64 s[60:61], s[56:57]
		*(p++) = 0xbe90010eu | (dst << 17);				// s_mov_b64 s[16 + dst * 2:17 + dst * 2], s[14:15]
#endif

		// 16 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_ISMULH_R;

	if (opcode < RANDOMX_FREQ_ISMULH_M)
	{
		if (prefetch_vgpr_index >= 0)
		{
			if (src != dst) // p = 7/8
				p = jit_scratchpad_calc_address(p, src, inst.y, (mod % 4) ? ScratchpadL1Mask_reg : ScratchpadL2Mask_reg, batch_size);
			else // p = 1/8
				p = jit_scratchpad_calc_fixed_address(p, inst.y & ScratchpadL3Mask, batch_size);

			p = jit_scratchpad_load(p, prefetch_vgpr_index ? prefetch_vgpr_index : 28);
		}

		if (prefetch_vgpr_index <= 0)
		{
			p = jit_scratchpad_load2(p, prefetch_vgpr_index ? -prefetch_vgpr_index : 28, prefetch_vgpr_index ? vmcnt : 0);

#if GCN_VERSION >= 15
			*(p++) = 0xbea60410u | (dst << 1);				// s_mov_b64 s[38:39], s[16 + dst * 2:17 + dst * 2]
			*(p++) = 0xbebc2138u;							// s_swappc_b64 s[60:61], s[56:57]
			*(p++) = 0xbe90040eu | (dst << 17);				// s_mov_b64 s[16 + dst * 2:17 + dst * 2], s[14:15]
#else
			*(p++) = 0xbea60110u | (dst << 1);				// s_mov_b64 s[38:39], s[16 + dst * 2:17 + dst * 2]
			*(p++) = 0xbebc1e38u;							// s_swappc_b64 s[60:61], s[56:57]
			*(p++) = 0xbe90010eu | (dst << 17);				// s_mov_b64 s[16 + dst * 2:17 + dst * 2], s[14:15]
#endif
		}

		// (12*7/8 + 8*1/8 + 28) + 12 = 51.5 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_ISMULH_M;

	if (opcode < RANDOMX_FREQ_IMUL_RCP)
	{
		if (inst.y & (inst.y - 1))
		{
			const uint2 rcp_value = as_uint2(imul_rcp_value(inst.y));

			*(p++) = S_MOV_B32_IMUL_RCP;					// s_mov_b32       s32, imm32
			*(p++) = rcp_value.x;
#if GCN_VERSION >= 14
			*(p++) = S_MUL_HI_U32_IMUL_RCP | (dst << 1);				// s_mul_hi_u32    s15, s(16 + dst * 2), s32
#else
			// v_mov_b32 v28, s32
			*(p++) = 0x7e380220u;
			// v_mul_hi_u32 v28, v28, s(16 + dst * 2)
			*(p++) = 0xd286001cu;
			*(p++) = 0x0000211cu + (dst << 10);
			// v_readlane_b32 s15, v28, 0
			*(p++) = 0xd289000fu;
			*(p++) = 0x0001011cu;
#endif
			*(p++) = S_MUL_I32_IMUL | 0x0eff10u | (dst << 1);				// s_mul_i32       s14, s(16 + dst * 2), imm32
			*(p++) = rcp_value.y;
			*(p++) = 0x800f0e0fu;							// s_add_u32       s15, s15, s14
			*(p++) = S_MUL_I32_IMUL | 0x0e2011u | (dst << 1);				// s_mul_i32       s14, s(17 + dst * 2), s32
			*(p++) = 0x80110e0fu | (dst << 17);				// s_add_u32       s(17 + dst * 2), s15, s14
			*(p++) = S_MUL_I32_IMUL | 0x102010u | (dst << 1) | (dst << 17);// s_mul_i32       s(16 + dst * 2), s(16 + dst * 2), s32
		}

		// 36 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_IMUL_RCP;

	if (opcode < RANDOMX_FREQ_INEG_R)
	{
		*(p++) = 0x80901080u | (dst << 9) | (dst << 17);	// s_sub_u32       s(16 + dst * 2), 0, s(16 + dst * 2)
		*(p++) = 0x82911180u | (dst << 9) | (dst << 17);	// s_subb_u32      s(17 + dst * 2), 0, s(17 + dst * 2)

		// 8 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_INEG_R;

	if (opcode < RANDOMX_FREQ_IXOR_R)
	{
		if (src != dst) // p = 7/8
		{
			// s_xor_b64 s[16 + dst * 2:17 + dst * 2], s[16 + dst * 2:17 + dst * 2], s[16 + src * 2:17 + src * 2]
			*(p++) = S_XOR_B32_64 | 0x901010u | (dst << 1) | (dst << 17) | (src << 9);
		}
		else // p = 1/8
		{
			if (as_int(inst.y) < 0) // p = 1/2
			{
				// s_mov_b32 s62, imm32
				*(p++) = S_MOV_B32_XOR_R;
				*(p++) = inst.y;

				// s_xor_b64 s[16 + dst * 2:17 + dst * 2], s[16 + dst * 2:17 + dst * 2], s[62:63]
				*(p++) = S_XOR_B32_64 | 0x903e10u | (dst << 1) | (dst << 17);
			}
			else
			{
				// s_xor_b32 s(16 + dst * 2), s(16 + dst * 2), imm32
				*(p++) = S_XOR_B32_64 | 0x10ff10u | (dst << 1) | (dst << 17);
				*(p++) = inst.y;
			}
		}

		// 4*7/8 + 12/16 + 8/16 = 4.75 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_IXOR_R;

	if (opcode < RANDOMX_FREQ_IXOR_M)
	{
		if (prefetch_vgpr_index >= 0)
		{
			if (src != dst) // p = 7/8
				p = jit_scratchpad_calc_address(p, src, inst.y, (mod % 4) ? ScratchpadL1Mask_reg : ScratchpadL2Mask_reg, batch_size);
			else // p = 1/8
				p = jit_scratchpad_calc_fixed_address(p, inst.y & ScratchpadL3Mask, batch_size);

			p = jit_scratchpad_load(p, prefetch_vgpr_index ? prefetch_vgpr_index : 28);
		}

		if (prefetch_vgpr_index <= 0)
		{
			p = jit_scratchpad_load2(p, prefetch_vgpr_index ? -prefetch_vgpr_index : 28, prefetch_vgpr_index ? vmcnt : 0);

			// s_xor_b64 s[16 + dst * 2:17 + dst * 2], s[16 + dst * 2:17 + dst * 2], s[14:15]
			*(p++) = S_XOR_B32_64 | 0x900e10u | (dst << 1) | (dst << 17);
		}

		// (12*7/8 + 8*1/8 + 28) + 4 = 43.5 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_IXOR_M;

	if (opcode < RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
	{
		if (src != dst) // p = 7/8
		{
			if (opcode < RANDOMX_FREQ_IROR_R)
			{
				// s_lshr_b64 s[32:33], s[16 + dst * 2:17 + dst * 2], s(16 + src * 2)
				*(p++) = S_LSHR | 0xa01010u | (dst << 1) | (src << 9);

				// s_sub_u32  s15, 64, s(16 + src * 2)
				*(p++) = 0x808f10c0u | (src << 9);

				// s_lshl_b64 s[34:35], s[16 + dst * 2:17 + dst * 2], s15
				*(p++) = S_LSHL | 0xa20f10u | (dst << 1);
			}
			else
			{
				// s_lshl_b64 s[32:33], s[16 + dst * 2:17 + dst * 2], s(16 + src * 2)
				*(p++) = S_LSHL | 0xa01010u | (dst << 1) | (src << 9);

				// s_sub_u32  s15, 64, s(16 + src * 2)
				*(p++) = 0x808f10c0u | (src << 9);

				// s_lshr_b64 s[34:35], s[16 + dst * 2:17 + dst * 2], s15
				*(p++) = S_LSHR | 0xa20f10u | (dst << 1);
			}
		}
		else // p = 1/8
		{
			const uint shift = ((opcode < RANDOMX_FREQ_IROR_R) ? inst.y : -inst.y) & 63;

			// s_lshr_b64 s[32:33], s[16 + dst * 2:17 + dst * 2], shift
			*(p++) = S_LSHR | 0xa08010u | (dst << 1) | (shift << 8);

			// s_lshl_b64 s[34:35], s[16 + dst * 2:17 + dst * 2], 64 - shift
			*(p++) = S_LSHL | 0xa28010u | (dst << 1) | ((64 - shift) << 8);
		}

		// s_or_b64 s[16 + dst * 2:17 + dst * 2], s[32:33], s[34:35]
		*(p++) = S_OR | 0x902220u | (dst << 17);

		// 12*7/8 + 8/8 + 4 = 15.5 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

	if (opcode < RANDOMX_FREQ_ISWAP_R)
	{
		if (src != dst)
		{
#if GCN_VERSION >= 15
			*(p++) = 0xbea00410u | (dst << 1);				// s_mov_b64       s[32:33], s[16 + dst * 2:17 + dst * 2]
			*(p++) = 0xbe900410u | (src << 1) | (dst << 17);// s_mov_b64       s[16 + dst * 2:17 + dst * 2], s[16 + src * 2:17 + src * 2]
			*(p++) = 0xbe900420u | (src << 17);				// s_mov_b64       s[16 + src * 2:17 + Src * 2], s[32:33]
#else
			*(p++) = 0xbea00110u | (dst << 1);				// s_mov_b64       s[32:33], s[16 + dst * 2:17 + dst * 2]
			*(p++) = 0xbe900110u | (src << 1) | (dst << 17);// s_mov_b64       s[16 + dst * 2:17 + dst * 2], s[16 + src * 2:17 + src * 2]
			*(p++) = 0xbe900120u | (src << 17);				// s_mov_b64       s[16 + src * 2:17 + Src * 2], s[32:33]
#endif
		}

		// 12*7/8 = 10.5 bytes on average
		return p;
	}
	opcode -= RANDOMX_FREQ_ISWAP_R;

	if (opcode < RANDOMX_FREQ_FSWAP_R)
	{
		// ds_swizzle_b32 v(60 + dst * 2), v(60 + dst * 2) offset:0x8001
		*(p++) = DS_SWIZZLE_B32_FSWAP_R;
		*(p++) = 0x3c00003cu + (dst << 1) + (dst << 25);

		// ds_swizzle_b32 v(61 + dst * 2), v(61 + dst * 2) offset:0x8001
		*(p++) = DS_SWIZZLE_B32_FSWAP_R;
		*(p++) = 0x3d00003du + (dst << 1) + (dst << 25);

		// s_waitcnt lgkmcnt(0)
		*(p++) = 0xbf8cc07fu;

		// 20 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_FSWAP_R;

	if (opcode < RANDOMX_FREQ_FADD_R)
	{
		// v_add_f64 v[60 + dst * 2:61 + dst * 2], v[60 + dst * 2:61 + dst * 2], v[52 + src * 2:53 + src * 2]
		*(p++) = V_ADD_F64 + ((dst & 3) << 1);
		*(p++) = 0x0002693cu + ((dst & 3) << 1) + ((src & 3) << 10);

		// 8 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_FADD_R;

	if (opcode < RANDOMX_FREQ_FADD_M)
	{
		if (prefetch_vgpr_index >= 0)
		{
			p = jit_scratchpad_calc_address_fp(p, src, inst.y, (mod % 4) ? ScratchpadL1Mask_reg : ScratchpadL2Mask_reg, batch_size);
			p = jit_scratchpad_load_fp(p, prefetch_vgpr_index ? prefetch_vgpr_index : 28);
		}

		if (prefetch_vgpr_index <= 0)
		{
			p = jit_scratchpad_load2_fp(p, prefetch_vgpr_index ? -prefetch_vgpr_index : 28, prefetch_vgpr_index ? vmcnt : 0);

			// v_add_f64 v[60 + dst * 2:61 + dst * 2], v[60 + dst * 2:61 + dst * 2], v[28:29]
			*(p++) = V_ADD_F64 + ((dst & 3) << 1);
			*(p++) = 0x0002393cu + ((dst & 3) << 1);
		}

		// 32 + 8 = 40 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_FADD_M;

	if (opcode < RANDOMX_FREQ_FSUB_R)
	{
		// v_add_f64 v[60 + dst * 2:61 + dst * 2], v[60 + dst * 2:61 + dst * 2], -v[52 + src * 2:53 + src * 2]
		*(p++) = V_ADD_F64 + ((dst & 3) << 1);
		*(p++) = 0x4002693cu + ((dst & 3) << 1) + ((src & 3) << 10);

		// 8 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_FSUB_R;

	if (opcode < RANDOMX_FREQ_FSUB_M)
	{
		if (prefetch_vgpr_index >= 0)
		{
			p = jit_scratchpad_calc_address_fp(p, src, inst.y, (mod % 4) ? ScratchpadL1Mask_reg : ScratchpadL2Mask_reg, batch_size);
			p = jit_scratchpad_load_fp(p, prefetch_vgpr_index ? prefetch_vgpr_index : 28);
		}

		if (prefetch_vgpr_index <= 0)
		{
			p = jit_scratchpad_load2_fp(p, prefetch_vgpr_index ? -prefetch_vgpr_index : 28, prefetch_vgpr_index ? vmcnt : 0);

			// v_add_f64 v[60 + dst * 2:61 + dst * 2], v[60 + dst * 2:61 + dst * 2], -v[28:29]
			*(p++) = V_ADD_F64 + ((dst & 3) << 1);
			*(p++) = 0x4002393cu + ((dst & 3) << 1);
		}

		// 32 + 8 = 40 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_FSUB_M;

	if (opcode < RANDOMX_FREQ_FSCAL_R)
	{
		// v_xor_b32 v(61 + dst * 2), v(61 + dst * 2), v51
		*(p++) = (V_XOR_B32 | 0x7a673du) + ((dst & 3) << 1) + ((dst & 3) << 18);

		// 4 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_FSCAL_R;

	if (opcode < RANDOMX_FREQ_FMUL_R)
	{
		// v_mul_f64 v[68 + dst * 2:69 + dst * 2], v[68 + dst * 2:69 + dst * 2], v[52 + src * 2:53 + src * 2]
		*(p++) = V_MUL_F64 + ((dst & 3) << 1);
		*(p++) = 0x00026944u + ((dst & 3) << 1) + ((src & 3) << 10);

		// 8 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_FMUL_R;

	if (opcode < RANDOMX_FREQ_FDIV_M)
	{
		if (prefetch_vgpr_index >= 0)
		{
			p = jit_scratchpad_calc_address_fp(p, src, inst.y, (mod % 4) ? ScratchpadL1Mask_reg : ScratchpadL2Mask_reg, batch_size);
			p = jit_scratchpad_load_fp(p, prefetch_vgpr_index ? prefetch_vgpr_index : 28);
		}

		if (prefetch_vgpr_index <= 0)
		{
			p = jit_scratchpad_load2_fp(p, prefetch_vgpr_index ? -prefetch_vgpr_index : 28, prefetch_vgpr_index ? vmcnt : 0);

			// s_swappc_b64 s[60:61], s[48 + dst * 2:49 + dst * 2]
#if GCN_VERSION >= 15
			*(p++) = 0xbebc2130u + ((dst & 3) << 1);
#else
			*(p++) = 0xbebc1e30u + ((dst & 3) << 1);
#endif
		}

		// 32 + 4 = 36 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_FDIV_M;

	if (opcode < RANDOMX_FREQ_FSQRT_R)
	{
		// s_swappc_b64 s[60:61], s[40 + dst * 2:41 + dst * 2]
#if GCN_VERSION >= 15
		*(p++) = 0xbebc2128u + ((dst & 3) << 1);
#else
		*(p++) = 0xbebc1e28u + ((dst & 3) << 1);
#endif

		// 4 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_FSQRT_R;

	if (opcode < RANDOMX_FREQ_CBRANCH)
	{
		const int shift = (mod >> 4) + RANDOMX_JUMP_OFFSET;
		uint imm = inst.y | (1u << shift);
		imm &= ~(1u << (shift - 1));

		// s_add_u32 s(16 + dst * 2), s(16 + dst * 2), imm32
		*(p++) = 0x8010ff10 | (dst << 1) | (dst << 17);
		*(p++) = imm;

		// s_addc_u32 s(17 + dst * 2), s(17 + dst * 2), ((imm < 0) ? -1 : 0)
		*(p++) = 0x82110011u | (dst << 1) | (dst << 17) | (((as_int(imm) < 0) ? 0xc1 : 0x80) << 8);

		const uint conditionMaskReg = 70 + (mod >> 4);

		// s_and_b32 s14, s(16 + dst * 2), conditionMaskReg
		*(p++) = S_AND | 0x0e0010u | (dst << 1) | (conditionMaskReg << 8);

		// s_cbranch_scc0 target
		const int delta = ((last_branch_target - p) - 1);
		*(p++) = 0xbf840000u | (delta & 0xFFFF);

		// 20 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_CBRANCH;

	if (opcode < RANDOMX_FREQ_CFROUND)
	{
		const uint shift = inst.y & 63;
		if (shift == 63)
		{
			*(p++) = S_LSHL | 0x0e8110u | (src << 1);		// s_lshl_b32      s14, s(16 + src * 2), 1
			*(p++) = S_LSHR | 0x0f9f11u | (src << 1);		// s_lshr_b32      s15, s(17 + src * 2), 31
			*(p++) = S_OR | 0x0e0f0eu;					// s_or_b32        s14, s14, s15
			*(p++) = S_AND | 0x0e830eu;					// s_and_b32       s14, s14, 3
		}
		else
		{
			// s_bfe_u64 s[14:15], s[16:17], (shift,width=2)
			*(p++) = S_BFE | 0x8eff10u | (src << 1);
			*(p++) = shift | (2 << 16);
		}

		// s_brev_b32 s14, s14
		// s_lshr_b32 s66, s14, 30
		// s_setreg_b32 hwreg(mode, 2, 2), s66
#if GCN_VERSION >= 15
		*(p++) = 0xbe8e0b0eu;
		*(p++) = 0x90429e0eu;
		*(p++) = 0xb9c20881u;
#else
		*(p++) = 0xbe8e080eu;
		*(p++) = 0x8f429e0eu;
		*(p++) = 0xb9420881u;
#endif

		// 20 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_CFROUND;

	if (opcode < RANDOMX_FREQ_ISTORE)
	{
		const uint mask = ((mod >> 4) < 14) ? ((mod % 4) ? ScratchpadL1Mask_reg : ScratchpadL2Mask_reg) : ScratchpadL3Mask_reg;
		p = jit_scratchpad_calc_address(p, dst, inst.y, mask, batch_size);

		const uint vgpr_id = 48;
		*(p++) = 0x7e000210u | (src << 1) | (vgpr_id << 17);	// v_mov_b32       vgpr_id, s(16 + src * 2)
		*(p++) = 0x7e020211u | (src << 1) | (vgpr_id << 17);	// v_mov_b32       vgpr_id + 1, s(17 + src * 2)

		// v28 = offset

#if GCN_VERSION >= 14
#if GCN_VERSION >= 15
		// s_waitcnt vmcnt(0)
		*(p++) = 0xbf8c3f70u;
#endif
		// global_store_dwordx2 v28, v[vgpr_id:vgpr_id + 1], s[0:1]
		*(p++) = 0xdc748000u;
		*(p++) = 0x0000001cu | (vgpr_id << 8);
#else
		// v_add_u32 v28, vcc, v28, v2
		*(p++) = 0x3238051cu;
		// v_addc_u32 v29, vcc, 0, v3, vcc
		*(p++) = 0x383a0680u;
		// flat_store_dwordx2 v[28:29], v[vgpr_id:vgpr_id + 1]
		*(p++) = 0xdc740000u;
		*(p++) = 0x0000001cu | (vgpr_id << 8);
#endif

		// 28 bytes
		return p;
	}
	opcode -= RANDOMX_FREQ_ISTORE;

	return p;
}

int jit_prefetch_read(
	__global uint2* p0,
	const int prefetch_data_count,
	const uint i,
	const uint src,
	const uint dst,
	const uint2 inst,
	const uint srcAvailableAt,
	const uint scratchpadAvailableAt,
	const uint scratchpadHighAvailableAt,
	const int lastBranchTarget,
	const int lastBranch)
{
	uint2 t;
	t.x = (src == dst) ? (((inst.y & ScratchpadL3Mask) >= RANDOMX_SCRATCHPAD_L2) ? scratchpadHighAvailableAt : scratchpadAvailableAt) : max(scratchpadAvailableAt, srcAvailableAt);
	t.y = i;

	const int t1 = t.x;

	if ((lastBranchTarget <= t1) && (t1 <= lastBranch))
	{
		// Don't move prefetch inside previous branch scope
		t.x = lastBranch + 1;
	}
	else if ((lastBranchTarget > lastBranch) && (t1 < lastBranchTarget))
	{
		// Don't move prefetch outside current branch scope
		t.x = lastBranchTarget;
	}

	p0[prefetch_data_count] = t;
	return prefetch_data_count + 1;
}

__global uint* generate_jit_code(__global uint2* e, __global uint2* p0, __global uint* p, uint batch_size)
{
	int prefetch_data_count;

	#pragma unroll 1
	for (volatile int pass = 0; pass < 2; ++pass)
	{
#if RANDOMX_PROGRAM_SIZE > 256
		int registerLastChanged[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
#else
		ulong registerLastChanged = 0;
		uint registerWasChanged = 0;
#endif

		uint scratchpadAvailableAt = 0;
		uint scratchpadHighAvailableAt = 0;

		int lastBranchTarget = -1;
		int lastBranch = -1;

#if RANDOMX_PROGRAM_SIZE > 256
		int registerLastChangedAtBranchTarget[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
#else
		ulong registerLastChangedAtBranchTarget = 0;
		uint registerWasChangedAtBranchTarget = 0;
#endif
		uint scratchpadAvailableAtBranchTarget = 0;
		uint scratchpadHighAvailableAtBranchTarget = 0;

		prefetch_data_count = 0;

		#pragma unroll 1
		for (uint i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
		{
			// Clean flags
			if (pass == 0)
				e[i].x &= ~(0xf8u << 8);

			uint2 inst = e[i];
			uint opcode = inst.x & 0xFF;
			const uint dst = (inst.x >> 8) & 7;
			const uint src = (inst.x >> 16) & 7;
			const uint mod = inst.x >> 24;

			if (pass == 1)
			{
				// Branch target
				if (inst.x & (0x20 << 8))
				{
 					lastBranchTarget = i;
#if RANDOMX_PROGRAM_SIZE > 256
					#pragma unroll
					for (int j = 0; j < 8; ++j)
						registerLastChangedAtBranchTarget[j] = registerLastChanged[j];
#else
					registerLastChangedAtBranchTarget = registerLastChanged;
					registerWasChangedAtBranchTarget = registerWasChanged;
#endif
					scratchpadAvailableAtBranchTarget = scratchpadAvailableAt;
					scratchpadHighAvailableAtBranchTarget = scratchpadHighAvailableAt;
				}

				// Branch
				if (inst.x & (0x40 << 8))
					lastBranch = i;
			}

#if RANDOMX_PROGRAM_SIZE > 256
			const uint srcAvailableAt = registerLastChanged[src] + 1;
			const uint dstAvailableAt = registerLastChanged[dst] + 1;
#else
			const uint srcAvailableAt = (registerWasChanged & (1u << src)) ? (((registerLastChanged >> (src * 8)) & 0xFF) + 1) : 0;
			const uint dstAvailableAt = (registerWasChanged & (1u << dst)) ? (((registerLastChanged >> (dst * 8)) & 0xFF) + 1) : 0;
#endif

			if (opcode < RANDOMX_FREQ_IADD_RS)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_RS;

			if (opcode < RANDOMX_FREQ_IADD_M)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				if (pass == 1)
					prefetch_data_count = jit_prefetch_read(p0, prefetch_data_count, i, src, dst, inst, srcAvailableAt, scratchpadAvailableAt, scratchpadHighAvailableAt, lastBranchTarget, lastBranch);
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_M;

			if (opcode < RANDOMX_FREQ_ISUB_R)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_ISUB_R;

			if (opcode < RANDOMX_FREQ_ISUB_M)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				if (pass == 1)
					prefetch_data_count = jit_prefetch_read(p0, prefetch_data_count, i, src, dst, inst, srcAvailableAt, scratchpadAvailableAt, scratchpadHighAvailableAt, lastBranchTarget, lastBranch);
				continue;
			}
			opcode -= RANDOMX_FREQ_ISUB_M;

			if (opcode < RANDOMX_FREQ_IMUL_R)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_R;

			if (opcode < RANDOMX_FREQ_IMUL_M)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				if (pass == 1)
					prefetch_data_count = jit_prefetch_read(p0, prefetch_data_count, i, src, dst, inst, srcAvailableAt, scratchpadAvailableAt, scratchpadHighAvailableAt, lastBranchTarget, lastBranch);
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_M;

			if (opcode < RANDOMX_FREQ_IMULH_R)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_IMULH_R;

			if (opcode < RANDOMX_FREQ_IMULH_M)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				if (pass == 1)
					prefetch_data_count = jit_prefetch_read(p0, prefetch_data_count, i, src, dst, inst, srcAvailableAt, scratchpadAvailableAt, scratchpadHighAvailableAt, lastBranchTarget, lastBranch);
				continue;
			}
			opcode -= RANDOMX_FREQ_IMULH_M;

			if (opcode < RANDOMX_FREQ_ISMULH_R)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_ISMULH_R;

			if (opcode < RANDOMX_FREQ_ISMULH_M)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				if (pass == 1)
					prefetch_data_count = jit_prefetch_read(p0, prefetch_data_count, i, src, dst, inst, srcAvailableAt, scratchpadAvailableAt, scratchpadHighAvailableAt, lastBranchTarget, lastBranch);
				continue;
			}
			opcode -= RANDOMX_FREQ_ISMULH_M;

			if (opcode < RANDOMX_FREQ_IMUL_RCP)
			{
				if (inst.y & (inst.y - 1))
				{
#if RANDOMX_PROGRAM_SIZE > 256
					registerLastChanged[dst] = i;
#else
					registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
					registerWasChanged |= 1u << dst;
#endif
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_RCP;

			if (opcode < RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R;

			if (opcode < RANDOMX_FREQ_IXOR_M)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				if (pass == 1)
					prefetch_data_count = jit_prefetch_read(p0, prefetch_data_count, i, src, dst, inst, srcAvailableAt, scratchpadAvailableAt, scratchpadHighAvailableAt, lastBranchTarget, lastBranch);
				continue;
			}
			opcode -= RANDOMX_FREQ_IXOR_M;

			if (opcode < RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
			{
#if RANDOMX_PROGRAM_SIZE > 256
				registerLastChanged[dst] = i;
#else
				registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
				registerWasChanged |= 1u << dst;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

			if (opcode < RANDOMX_FREQ_ISWAP_R)
			{
				if (src != dst)
				{
#if RANDOMX_PROGRAM_SIZE > 256
					registerLastChanged[dst] = i;
					registerLastChanged[src] = i;
#else
					registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
					registerLastChanged = (registerLastChanged & ~(0xFFul << (src * 8))) | ((ulong)(i) << (src * 8));
					registerWasChanged |= (1u << dst) | (1u << src);
#endif
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_ISWAP_R;

			if (opcode < RANDOMX_FREQ_FSWAP_R + RANDOMX_FREQ_FADD_R)
			{
				continue;
			}
			opcode -= RANDOMX_FREQ_FSWAP_R + RANDOMX_FREQ_FADD_R;

			if (opcode < RANDOMX_FREQ_FADD_M)
			{
				if (pass == 1)
					prefetch_data_count = jit_prefetch_read(p0, prefetch_data_count, i, src, 0xFF, inst, srcAvailableAt, scratchpadAvailableAt, scratchpadHighAvailableAt, lastBranchTarget, lastBranch);
				continue;
			}
			opcode -= RANDOMX_FREQ_FADD_M;

			if (opcode < RANDOMX_FREQ_FSUB_R)
			{
				continue;
			}
			opcode -= RANDOMX_FREQ_FSUB_R;

			if (opcode < RANDOMX_FREQ_FSUB_M)
			{
				if (pass == 1)
					prefetch_data_count = jit_prefetch_read(p0, prefetch_data_count, i, src, 0xFF, inst, srcAvailableAt, scratchpadAvailableAt, scratchpadHighAvailableAt, lastBranchTarget, lastBranch);
				continue;
			}
			opcode -= RANDOMX_FREQ_FSUB_M;

			if (opcode < RANDOMX_FREQ_FSCAL_R + RANDOMX_FREQ_FMUL_R)
			{
				continue;
			}
			opcode -= RANDOMX_FREQ_FSCAL_R + RANDOMX_FREQ_FMUL_R;

			if (opcode < RANDOMX_FREQ_FDIV_M)
			{
				if (pass == 1)
					prefetch_data_count = jit_prefetch_read(p0, prefetch_data_count, i, src, 0xFF, inst, srcAvailableAt, scratchpadAvailableAt, scratchpadHighAvailableAt, lastBranchTarget, lastBranch);
				continue;
			}
			opcode -= RANDOMX_FREQ_FDIV_M;

			if (opcode < RANDOMX_FREQ_FSQRT_R)
			{
				continue;
			}
			opcode -= RANDOMX_FREQ_FSQRT_R;

			if (opcode < RANDOMX_FREQ_CBRANCH)
			{
				if (pass == 0)
				{
					// Workaround for a bug in AMD 18.6.1 driver
					volatile uint dstAvailableAt2 = dstAvailableAt;

					// Mark branch target
					e[dstAvailableAt2].x |= (0x20 << 8);

					// Mark branch
					e[i].x |= (0x40 << 8);

					// Set all registers as changed at this instruction as per RandomX specification
#if RANDOMX_PROGRAM_SIZE > 256
					#pragma unroll
					for (int j = 0; j < 8; ++j)
						registerLastChanged[j] = i;
#else
					uint t = i | (i << 8);
					t = t | (t << 16);
					registerLastChanged = t;
					registerLastChanged = registerLastChanged | (registerLastChanged << 32);
					registerWasChanged = 0xFF;
#endif
				}
				else
				{
					// Update only registers which really changed inside this branch
#if RANDOMX_PROGRAM_SIZE > 256
					registerLastChanged[dst] = i;
#else
					registerLastChanged = (registerLastChanged & ~(0xFFul << (dst * 8))) | ((ulong)(i) << (dst * 8));
					registerWasChanged |= 1u << dst;
#endif

					for (int reg = 0; reg < 8; ++reg)
					{
#if RANDOMX_PROGRAM_SIZE > 256
						const uint availableAtBranchTarget = registerLastChangedAtBranchTarget[reg] + 1;
						const uint availableAt = registerLastChanged[reg] + 1;
						if (availableAt != availableAtBranchTarget)
						{
							registerLastChanged[reg] = i;
						}
#else
						const uint availableAtBranchTarget = (registerWasChangedAtBranchTarget & (1u << reg)) ? (((registerLastChangedAtBranchTarget >> (reg * 8)) & 0xFF) + 1) : 0;
						const uint availableAt = (registerWasChanged & (1u << reg)) ? (((registerLastChanged >> (reg * 8)) & 0xFF) + 1) : 0;
						if (availableAt != availableAtBranchTarget)
						{
							registerLastChanged = (registerLastChanged & ~(0xFFul << (reg * 8))) | ((ulong)(i) << (reg * 8));
							registerWasChanged |= 1u << reg;
						}
#endif
					}

					if (scratchpadAvailableAtBranchTarget != scratchpadAvailableAt)
						scratchpadAvailableAt = i + 1;

					if (scratchpadHighAvailableAtBranchTarget != scratchpadHighAvailableAt)
						scratchpadHighAvailableAt = i + 1;
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_CBRANCH;

			if (opcode < RANDOMX_FREQ_CFROUND)
			{
				continue;
			}
			opcode -= RANDOMX_FREQ_CFROUND;

			if (opcode < RANDOMX_FREQ_ISTORE)
			{
				if (pass == 0)
				{
					// Mark ISTORE
					e[i].x = inst.x | (0x80 << 8);
				}
				else
				{
					scratchpadAvailableAt = i + 1;
					if ((mod >> 4) >= 14)
						scratchpadHighAvailableAt = i + 1;
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_ISTORE;
		}
	}

	// Sort p0
	uint prev = p0[0].x;
	#pragma unroll 1
	for (int j = 1; j < prefetch_data_count; ++j)
	{
		uint2 cur = p0[j];
		if (cur.x >= prev)
		{
			prev = cur.x;
			continue;
		}

		int j1 = j - 1;
		do {
			p0[j1 + 1] = p0[j1];
			--j1;
		} while ((j1 >= 0) && (p0[j1].x >= cur.x));
		p0[j1 + 1] = cur;
	}
	p0[prefetch_data_count].x = RANDOMX_PROGRAM_SIZE;

	__global int* prefecth_vgprs_stack = (__global int*)(p0 + prefetch_data_count + 1);

	// v86 - v127 will be used for global memory loads
	enum { num_prefetch_vgprs = 21 };

	#pragma unroll
	for (int i = 0; i < num_prefetch_vgprs; ++i)
		prefecth_vgprs_stack[i] = NUM_VGPR_REGISTERS - 2 - i * 2;

	__global int* prefetched_vgprs = prefecth_vgprs_stack + num_prefetch_vgprs;

	#pragma unroll 8
	for (int i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
		prefetched_vgprs[i] = 0;

	int k = 0;
	uint2 prefetch_data = p0[0];
	int mem_counter = 0;
	int s_waitcnt_value = 63;
	int num_prefetch_vgprs_available = num_prefetch_vgprs;

	__global uint* last_branch_target = p;

	const uint size_limit = (COMPILED_PROGRAM_SIZE - 200) / sizeof(uint);
	__global uint* start_p = p;

	#pragma unroll 1
	for (int i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
	{
		const uint2 inst = e[i];

		if (inst.x & (0x20 << 8))
			last_branch_target = p;

		bool done = false;
		do {
			uint2 jit_inst;
			int jit_prefetch_vgpr_index;
			int jit_vmcnt;

			if (!done && (prefetch_data.x == i) && (num_prefetch_vgprs_available > 0))
			{
				++mem_counter;
				const int vgpr_id = prefecth_vgprs_stack[--num_prefetch_vgprs_available];
				prefetched_vgprs[prefetch_data.y] = vgpr_id | (mem_counter << 16);

				jit_inst = e[prefetch_data.y];
				jit_prefetch_vgpr_index = vgpr_id;
				jit_vmcnt = mem_counter;

				s_waitcnt_value = 63;

				++k;
				prefetch_data = p0[k];
			}
			else
			{
				const int prefetched_vgprs_data = prefetched_vgprs[i];
				const int vgpr_id = prefetched_vgprs_data & 0xFFFF;
				const int prev_mem_counter = prefetched_vgprs_data >> 16;
				if (vgpr_id)
					prefecth_vgprs_stack[num_prefetch_vgprs_available++] = vgpr_id;

				if (inst.x & (0x80 << 8))
				{
					++mem_counter;
					s_waitcnt_value = 63;
				}

				const int vmcnt = mem_counter - prev_mem_counter;

				jit_inst = inst;
				jit_prefetch_vgpr_index = -vgpr_id;
				jit_vmcnt = (vmcnt < s_waitcnt_value) ? vmcnt : -1;

				if (vmcnt < s_waitcnt_value)
					s_waitcnt_value = vmcnt;

				done = true;
			}

			p = jit_emit_instruction(p, last_branch_target, jit_inst, jit_prefetch_vgpr_index, jit_vmcnt, batch_size);
			if (p - start_p > size_limit)
			{
				// Code size limit exceeded!!!
				// Jump back to randomx_run kernel
				*(p++) = S_SETPC_B64_S12_13; // s_setpc_b64 s[12:13]
				return p;
			}
		} while (!done);
	}

	// Jump back to randomx_run kernel
	*(p++) = S_SETPC_B64_S12_13; // s_setpc_b64 s[12:13]
	return p;
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void randomx_jit(__global ulong* entropy, __global ulong* registers, __global uint2* intermediate_programs, __global uint* programs, uint batch_size, __global uint32_t* rounding, uint32_t iteration)
{
	const uint global_index = get_global_id(0) / 32;
	const uint sub = get_global_id(0) % 32;

	if (sub != 0)
		return;

	__global uint2* e = (__global uint2*)(entropy + global_index * (ENTROPY_SIZE / sizeof(ulong)) + (128 / sizeof(ulong)));
	__global uint2* p0 = intermediate_programs + global_index * (INTERMEDIATE_PROGRAM_SIZE / sizeof(uint2));
	__global uint* p = programs + global_index * (COMPILED_PROGRAM_SIZE / sizeof(uint));

	generate_jit_code(e, p0, p, batch_size);

	if (iteration == 0)
		rounding[global_index] = 0;

	__global ulong* R = registers + global_index * 32;
	entropy += global_index * (ENTROPY_SIZE / sizeof(ulong));

	// Group R registers
	R[0] = 0;
	R[1] = 0;
	R[2] = 0;
	R[3] = 0;
	R[4] = 0;
	R[5] = 0;
	R[6] = 0;
	R[7] = 0;

	// Group A registers
	__global double* A = (__global double*)(R + 24);
	A[0] = getSmallPositiveFloatBits(entropy[0]);
	A[1] = getSmallPositiveFloatBits(entropy[1]);
	A[2] = getSmallPositiveFloatBits(entropy[2]);
	A[3] = getSmallPositiveFloatBits(entropy[3]);
	A[4] = getSmallPositiveFloatBits(entropy[4]);
	A[5] = getSmallPositiveFloatBits(entropy[5]);
	A[6] = getSmallPositiveFloatBits(entropy[6]);
	A[7] = getSmallPositiveFloatBits(entropy[7]);

	// ma, mx
	((__global uint*)(R + 16))[0] = entropy[8] & CacheLineAlignMask;
	((__global uint*)(R + 16))[1] = entropy[10];

	// address registers
	uint addressRegisters = entropy[12];
	((__global uint*)(R + 17))[0] = 0 + (addressRegisters & 1);
	addressRegisters >>= 1;
	((__global uint*)(R + 17))[1] = 2 + (addressRegisters & 1);
	addressRegisters >>= 1;
	((__global uint*)(R + 17))[2] = 4 + (addressRegisters & 1);
	addressRegisters >>= 1;
	((__global uint*)(R + 17))[3] = 6 + (addressRegisters & 1);

	// dataset offset
	((__global uint*)(R + 19))[0] = (entropy[13] & DatasetExtraItems) * CacheLineSize;

	// eMask
	R[20] = getFloatMask(entropy[14]);
	R[21] = getFloatMask(entropy[15]);
}
