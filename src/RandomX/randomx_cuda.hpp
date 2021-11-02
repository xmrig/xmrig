#pragma once

/*
Copyright (c) 2019 SChernykh
Portions Copyright (c) 2018-2019 tevador

This file is part of RandomX CUDA.

RandomX CUDA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX CUDA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX CUDA.  If not, see<http://www.gnu.org/licenses/>.
*/

constexpr size_t HASH_SIZE = 64;
constexpr size_t ENTROPY_SIZE = 128 + ((RANDOMX_PROGRAM_SIZE * 8 + 127) / 128) * 128;
constexpr size_t REGISTERS_SIZE = 256;
constexpr size_t IMM_BUF_SIZE = RANDOMX_PROGRAM_SIZE * 4 - REGISTERS_SIZE;
constexpr size_t IMM_INDEX_COUNT = (IMM_BUF_SIZE / 4) - 2;
constexpr size_t VM_STATE_SIZE = REGISTERS_SIZE + IMM_BUF_SIZE + RANDOMX_PROGRAM_SIZE * 4;

constexpr uint32_t CacheLineSize = 64;
constexpr int ScratchpadL3Mask64 = RANDOMX_SCRATCHPAD_L3 - CacheLineSize;
constexpr uint32_t CacheLineAlignMask = (RANDOMX_DATASET_BASE_SIZE - 1) & ~(CacheLineSize - 1);

__device__ double getSmallPositiveFloatBits(uint64_t entropy)
{
	auto exponent = entropy >> 59; //0..31
	auto mantissa = entropy & randomx::mantissaMask;
	exponent += randomx::exponentBias;
	exponent &= randomx::exponentMask;
	exponent <<= randomx::mantissaSize;
	return __longlong_as_double(exponent | mantissa);
}

__device__ uint64_t getStaticExponent(uint64_t entropy)
{
	auto exponent = randomx::constExponentBits;
	exponent |= (entropy >> (64 - randomx::staticExponentBits)) << randomx::dynamicExponentBits;
	exponent <<= randomx::mantissaSize;
	return exponent;
}

__device__ uint64_t getFloatMask(uint64_t entropy)
{
	constexpr uint64_t mask22bit = (1ULL << 22) - 1;
	return (entropy & mask22bit) | getStaticExponent(entropy);
}

template<typename T>
__device__ T bit_cast(double value)
{
	return static_cast<T>(__double_as_longlong(value));
}

__device__ double load_F_E_groups(int value, uint64_t andMask, uint64_t orMask)
{
	uint64_t x = bit_cast<uint64_t>(__int2double_rn(value));
	x &= andMask;
	x |= orMask;
	return __longlong_as_double(static_cast<int64_t>(x));
}

__device__ void test_memory_access(uint64_t* r, uint8_t* scratchpad, uint32_t batch_size)
{
	uint32_t x = static_cast<uint32_t>(r[0]);
	uint64_t y = r[1];

	#pragma unroll
	for (int i = 0; i < 55; ++i)
	{
		x = x * 0x08088405U + 1;

		uint32_t mask = 16320;
		if (x < 0x58000000U) mask = 262080;
		if (x < 0x20000000U) mask = 2097088;

		uint32_t addr = x & mask;
		uint64_t offset;
		asm("mul.wide.u32 %0,%1,%2;" : "=l"(offset) : "r"(addr), "r"(batch_size));

		x = x * 0x08088405U + 1;
		uint64_t* p = (uint64_t*)(scratchpad + offset + (x & 56));
		if (x <= 3045522264U)
			y ^= *p; // 39/55
		else
			*p = y; // 16/55
	}

	r[1] = y;
}

//
// VM state:
//
// Bytes 0-255: registers
// Bytes 256-1023: imm32 values (up to 192 values can be stored). IMUL_RCP and CBRANCH use 2 consecutive imm32 values.
// Bytes 1024-2047: up to 256 instructions
//
// Instruction encoding:
//
// Bits 0-2: dst (0-7)
// Bits 3-5: src (0-7)
// Bits 6-13: imm32/64 offset (in DWORDs, 0-191)
// Bit 14: src location (register, scratchpad)
// Bits 15-16: src shift (0-3), ADD/MUL switch for FMA instruction
// Bit 17: src=imm32
// Bit 18: src=imm64
// Bit 19: src = -src
// Bits 20-23: opcode (add_rs, add, mul, umul_hi, imul_hi, neg, xor, ror, swap, cbranch, store, fswap, fma, fsqrt, fdiv, cfround)
// Bits 24-27: how many parallel instructions to run starting with this one (1-16)
// Bits 28-31: how many of them are FP instructions (0-8)
//

#define DST_OFFSET			0
#define SRC_OFFSET			3
#define IMM_OFFSET			6
#define LOC_OFFSET			14
#define SHIFT_OFFSET		15
#define SRC_IS_IMM32_OFFSET	17
#define SRC_IS_IMM64_OFFSET	18
#define NEGATIVE_SRC_OFFSET	19
#define OPCODE_OFFSET		20
#define NUM_INSTS_OFFSET	24
#define NUM_FP_INSTS_OFFSET	28

// ISWAP r0, r0
#define INST_NOP			(8 << OPCODE_OFFSET)

template<size_t N> struct get_power_of_2 { enum { Value = get_power_of_2<N / 2>::Value + 1 }; };
template<> struct get_power_of_2<1> { enum { Value = 0 }; };

#define LOC_L1 (32 - get_power_of_2<RANDOMX_SCRATCHPAD_L1>::Value)
#define LOC_L2 (32 - get_power_of_2<RANDOMX_SCRATCHPAD_L2>::Value)
#define LOC_L3 (32 - get_power_of_2<RANDOMX_SCRATCHPAD_L3>::Value)

__device__ uint64_t imul_rcp_value(uint32_t divisor)
{
	if ((divisor & (divisor - 1)) == 0)
	{
		return 1ULL;
	}

	const uint64_t p2exp63 = 1ULL << 63;

	uint64_t quotient = p2exp63 / divisor;
	uint64_t remainder = p2exp63 % divisor;

	uint32_t bsr;
	asm("bfind.u32 %0,%1;" : "=r"(bsr) : "r"(divisor));

	for (uint32_t shift = 0; shift <= bsr; ++shift)
	{
		const bool b = (remainder >= divisor - remainder);
		quotient = (quotient << 1) | (b ? 1 : 0);
		remainder = (remainder << 1) - (b ? divisor : 0);
	}

	return quotient;
}

__device__ void set_byte(uint64_t& a, uint32_t position, uint64_t value)
{
	asm("bfi.b64 %0,%1,%2,%3,8;" : "=l"(a) : "l"(value), "l"(a), "r"(position << 3));
}

__device__ uint32_t get_byte(uint64_t a, uint32_t position)
{
	uint64_t result;
	asm("bfe.u64 %0,%1,%2,8;" : "=l"(result) : "l"(a), "r"(position * 8));
	return static_cast<uint32_t>(result);
}

template<typename T, typename U, size_t N>
__device__ void set_buffer(T (&dst_buf)[N], const U value)
{
	uint32_t i = threadIdx.x * sizeof(T);
	const uint32_t step = blockDim.x * sizeof(T);
	uint8_t* dst = ((uint8_t*) dst_buf) + i;
	while (i < sizeof(T) * N)
	{
		*(T*)(dst) = static_cast<T>(value);
		dst += step;
		i += step;
	}
}

template<typename T, typename U>
__device__ void update_max(T& value, const U next_value)
{
	if (value < next_value)
		value = static_cast<T>(next_value);
}

__device__ void print_inst(uint2 inst)
{
	uint32_t opcode = inst.x & 0xff;
	const uint32_t dst = (inst.x >> 8) & 7;
	const uint32_t src = (inst.x >> 16) & 7;
	const uint32_t mod = (inst.x >> 24);
	const char* location = (src == dst) ? "L3" : ((mod % 4) ? "L1" : "L2");
	const char* branch_target = ((inst.x & (0x40 << 8)) != 0) ? "*" : (((inst.x & (0x10 << 8)) != 0) ? "!" : " ");
	const char* fp_inst = ((inst.x & (0x20 << 8)) != 0) ? "^" : " ";

	do {
		if (opcode < RANDOMX_FREQ_IADD_RS)
		{
			printf("%s%sIADD_RS  r%u, r%u    ", branch_target, fp_inst, dst, src);
			break;
		}
		opcode -= RANDOMX_FREQ_IADD_RS;

		if (opcode < RANDOMX_FREQ_IADD_M)
		{
			printf("%s%sIADD_M   r%u, %s[r%u]", branch_target, fp_inst, dst, location, src);
			break;
		}
		opcode -= RANDOMX_FREQ_IADD_M;

		if (opcode < RANDOMX_FREQ_ISUB_R)
		{
			printf("%s%sISUB_R   r%u, r%u    ", branch_target, fp_inst, dst, src);
			break;
		}
		opcode -= RANDOMX_FREQ_ISUB_R;

		if (opcode < RANDOMX_FREQ_ISUB_M)
		{
			printf("%s%sISUB_M   r%u, %s[r%u]", branch_target, fp_inst, dst, location, src);
			break;
		}
		opcode -= RANDOMX_FREQ_ISUB_M;

		if (opcode < RANDOMX_FREQ_IMUL_R)
		{
			printf("%s%sIMUL_R   r%u, r%u    ", branch_target, fp_inst, dst, src);
			break;
		}
		opcode -= RANDOMX_FREQ_IMUL_R;

		if (opcode < RANDOMX_FREQ_IMUL_M)
		{
			printf("%s%sIMUL_M   r%u, %s[r%u]", branch_target, fp_inst, dst, location, src);
			break;
		}
		opcode -= RANDOMX_FREQ_IMUL_M;

		if (opcode < RANDOMX_FREQ_IMULH_R)
		{
			printf("%s%sIMULH_R  r%u, r%u    ", branch_target, fp_inst, dst, src);
			break;
		}
		opcode -= RANDOMX_FREQ_IMULH_R;

		if (opcode < RANDOMX_FREQ_IMULH_M)
		{
			printf("%s%sIMULH_M  r%u, %s[r%u]", branch_target, fp_inst, dst, location, src);
			break;
		}
		opcode -= RANDOMX_FREQ_IMULH_M;

		if (opcode < RANDOMX_FREQ_ISMULH_R)
		{
			printf("%s%sISMULH_R r%u, r%u    ", branch_target, fp_inst, dst, src);
			break;
		}
		opcode -= RANDOMX_FREQ_ISMULH_R;

		if (opcode < RANDOMX_FREQ_ISMULH_M)
		{
			printf("%s%sISMULH_M r%u, %s[r%u]", branch_target, fp_inst, dst, location, src);
			break;
		}
		opcode -= RANDOMX_FREQ_ISMULH_M;

		if (opcode < RANDOMX_FREQ_IMUL_RCP)
		{
			printf("%s%sIMUL_RCP r%u        ", branch_target, fp_inst, dst);
			break;
		}
		opcode -= RANDOMX_FREQ_IMUL_RCP;

		if (opcode < RANDOMX_FREQ_INEG_R)
		{
			printf("%s%sINEG_R   r%u        ", branch_target, fp_inst, dst);
			break;
		}
		opcode -= RANDOMX_FREQ_INEG_R;

		if (opcode < RANDOMX_FREQ_IXOR_R)
		{
			printf("%s%sIXOR_R   r%u, r%u    ", branch_target, fp_inst, dst, src);
			break;
		}
		opcode -= RANDOMX_FREQ_IXOR_R;

		if (opcode < RANDOMX_FREQ_IXOR_M)
		{
			printf("%s%sIXOR_M   r%u, %s[r%u]", branch_target, fp_inst, dst, location, src);
			break;
		}
		opcode -= RANDOMX_FREQ_IXOR_M;

		if (opcode < RANDOMX_FREQ_IROR_R)
		{
			printf("%s%sIROR_R   r%u, r%u    ", branch_target, fp_inst, dst, src);
			break;
		}
		opcode -= RANDOMX_FREQ_IROR_R;

		if (opcode < RANDOMX_FREQ_IROL_R)
		{
			printf("%s%sIROL_R   r%u, r%u    ", branch_target, fp_inst, dst, src);
			break;
		}
		opcode -= RANDOMX_FREQ_IROL_R;

		if (opcode < RANDOMX_FREQ_ISWAP_R)
		{
			printf("%s%sISWAP_R  r%u, r%u    ", branch_target, fp_inst, dst, src);
			break;
		}
		opcode -= RANDOMX_FREQ_ISWAP_R;

		if (opcode < RANDOMX_FREQ_FSWAP_R)
		{
			printf("%s%sFSWAP_R  %s%u        ", branch_target, fp_inst, (dst < randomx::RegisterCountFlt) ? "f" : "e", dst % randomx::RegisterCountFlt);
			break;
		}
		opcode -= RANDOMX_FREQ_FSWAP_R;

		if (opcode < RANDOMX_FREQ_FADD_R)
		{
			printf("%s%sFADD_R   f%u, a%u    ", branch_target, fp_inst, dst % randomx::RegisterCountFlt, src % randomx::RegisterCountFlt);
			break;
		}
		opcode -= RANDOMX_FREQ_FADD_R;

		if (opcode < RANDOMX_FREQ_FADD_M)
		{
			printf("%s%sFADD_M   f%u, %s[r%u]", branch_target, fp_inst, dst % randomx::RegisterCountFlt, location, src);
			break;
		}
		opcode -= RANDOMX_FREQ_FADD_M;

		if (opcode < RANDOMX_FREQ_FSUB_R)
		{
			printf("%s%sFSUB_R   f%u, a%u    ", branch_target, fp_inst, dst % randomx::RegisterCountFlt, src % randomx::RegisterCountFlt);
			break;
		}
		opcode -= RANDOMX_FREQ_FSUB_R;

		if (opcode < RANDOMX_FREQ_FSUB_M)
		{
			printf("%s%sFSUB_M   f%u, %s[r%u]", branch_target, fp_inst, dst % randomx::RegisterCountFlt, location, src);
			break;
		}
		opcode -= RANDOMX_FREQ_FSUB_M;

		if (opcode < RANDOMX_FREQ_FSCAL_R)
		{
			printf("%s%sFSCAL_R  f%u        ", branch_target, fp_inst, dst % randomx::RegisterCountFlt);
			break;
		}
		opcode -= RANDOMX_FREQ_FSCAL_R;

		if (opcode < RANDOMX_FREQ_FMUL_R)
		{
			printf("%s%sFMUL_R   e%u, a%u    ", branch_target, fp_inst, dst % randomx::RegisterCountFlt, src % randomx::RegisterCountFlt);
			break;
		}
		opcode -= RANDOMX_FREQ_FMUL_R;

		if (opcode < RANDOMX_FREQ_FDIV_M)
		{
			printf("%s%sFDIV_M   e%u, %s[r%u]", branch_target, fp_inst, dst % randomx::RegisterCountFlt, location, src);
			break;
		}
		opcode -= RANDOMX_FREQ_FDIV_M;

		if (opcode < RANDOMX_FREQ_FSQRT_R)
		{
			printf("%s%sFSQRT_R  e%u        ", branch_target, fp_inst, dst % randomx::RegisterCountFlt);
			break;
		}
		opcode -= RANDOMX_FREQ_FSQRT_R;

		if (opcode < RANDOMX_FREQ_CBRANCH)
		{
			const int32_t lastChanged = (inst.x & (0x80 << 8)) ? -1 : static_cast<int32_t>((inst.x >> 16) & 0xFF);
			printf("%s%sCBRANCH  r%u, %3d   ", branch_target, fp_inst, dst, lastChanged + 1);
			break;
		}
		opcode -= RANDOMX_FREQ_CBRANCH;

		if (opcode < RANDOMX_FREQ_CFROUND)
		{
			printf("%s%sCFROUND  r%u, %2d    ", branch_target, fp_inst, src, inst.y & 63);
			break;
		}
		opcode -= RANDOMX_FREQ_CFROUND;

		if (opcode < RANDOMX_FREQ_ISTORE)
		{
			location = ((mod >> 4) >= randomx::StoreL3Condition) ? "L3" : ((mod % 4) ? "L1" : "L2");
			printf("%s%sISTORE   %s[r%u], r%u", branch_target, fp_inst, location, dst, src);
			break;
		}
		opcode -= RANDOMX_FREQ_ISTORE;

		printf("%s%sNOP%03u   r%u, r%u    ", branch_target, fp_inst, opcode, dst, src);
	} while (false);
}

template<int WORKERS_PER_HASH>
__global__ void __launch_bounds__(32, 16) init_vm(void* entropy_data, void* vm_states)
{
#if RANDOMX_PROGRAM_SIZE <= 256
	typedef uint8_t exec_t;
#else
	typedef uint16_t exec_t;
#endif

	__shared__ uint32_t execution_plan_buf[RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH * (32 / 8) * sizeof(exec_t) / sizeof(uint32_t)];

	set_buffer(execution_plan_buf, 0);

#	if (__CUDACC_VER_MAJOR__ >= 9)
	__syncwarp();
#	else
	__syncthreads();
#	endif

	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx = global_index / 8;
	const uint32_t sub = global_index % 8;

	exec_t* execution_plan = (exec_t*)(execution_plan_buf + (threadIdx.x / 8) * RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH * sizeof(exec_t) / sizeof(uint32_t));

	uint64_t* R = ((uint64_t*) vm_states) + idx * VM_STATE_SIZE / sizeof(uint64_t);
	R[sub] = 0;

	const uint64_t* entropy = ((const uint64_t*) entropy_data) + idx * ENTROPY_SIZE / sizeof(uint64_t);

	double* A = (double*)(R + 24);
	A[sub] = getSmallPositiveFloatBits(entropy[sub]);

	if (sub == 0)
	{
		uint2* src_program = (uint2*)(entropy + 128 / sizeof(uint64_t));

#if RANDOMX_PROGRAM_SIZE <= 256
		uint64_t registerLastChanged = 0;
		uint64_t registerWasChanged = 0;
#else
		int32_t registerLastChanged[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
#endif

		// Initialize CBRANCH instructions
		for (uint32_t i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
		{
			// Clear all src flags (branch target, FP, branch)
			*(uint32_t*)(src_program + i) &= ~(0xF8U << 8);

			const uint2 src_inst = src_program[i];
			uint2 inst = src_inst;

			uint32_t opcode = inst.x & 0xff;
			const uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;

			if (opcode < RANDOMX_FREQ_IADD_RS + RANDOMX_FREQ_IADD_M + RANDOMX_FREQ_ISUB_R + RANDOMX_FREQ_ISUB_M + RANDOMX_FREQ_IMUL_R + RANDOMX_FREQ_IMUL_M + RANDOMX_FREQ_IMULH_R + RANDOMX_FREQ_IMULH_M + RANDOMX_FREQ_ISMULH_R + RANDOMX_FREQ_ISMULH_M)
			{
#if RANDOMX_PROGRAM_SIZE <= 256
				set_byte(registerLastChanged, dst, i);
				set_byte(registerWasChanged, dst, 1);
#else
				registerLastChanged[dst] = i;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_RS + RANDOMX_FREQ_IADD_M + RANDOMX_FREQ_ISUB_R + RANDOMX_FREQ_ISUB_M + RANDOMX_FREQ_IMUL_R + RANDOMX_FREQ_IMUL_M + RANDOMX_FREQ_IMULH_R + RANDOMX_FREQ_IMULH_M + RANDOMX_FREQ_ISMULH_R + RANDOMX_FREQ_ISMULH_M;

			if (opcode < RANDOMX_FREQ_IMUL_RCP)
			{
				if (inst.y & (inst.y - 1))
				{
#if RANDOMX_PROGRAM_SIZE <= 256
					set_byte(registerLastChanged, dst, i);
					set_byte(registerWasChanged, dst, 1);
#else
					registerLastChanged[dst] = i;
#endif
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_RCP;

			if (opcode < RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R + RANDOMX_FREQ_IXOR_M + RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
			{
#if RANDOMX_PROGRAM_SIZE <= 256
				set_byte(registerLastChanged, dst, i);
				set_byte(registerWasChanged, dst, 1);
#else
				registerLastChanged[dst] = i;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R + RANDOMX_FREQ_IXOR_M + RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

			if (opcode < RANDOMX_FREQ_ISWAP_R)
			{
				if (src != dst)
				{
#if RANDOMX_PROGRAM_SIZE <= 256
					set_byte(registerLastChanged, dst, i);
					set_byte(registerWasChanged, dst, 1);
					set_byte(registerLastChanged, src, i);
					set_byte(registerWasChanged, src, 1);
#else
					registerLastChanged[dst] = i;
					registerLastChanged[src] = i;
#endif
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_ISWAP_R;

			if (opcode < RANDOMX_FREQ_FSWAP_R + RANDOMX_FREQ_FADD_R + RANDOMX_FREQ_FADD_M + RANDOMX_FREQ_FSUB_R + RANDOMX_FREQ_FSUB_M + RANDOMX_FREQ_FSCAL_R + RANDOMX_FREQ_FMUL_R + RANDOMX_FREQ_FDIV_M + RANDOMX_FREQ_FSQRT_R)
			{
				// Mark FP instruction (src |= 0x20)
				*(uint32_t*)(src_program + i) |= 0x20 << 8;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSWAP_R + RANDOMX_FREQ_FADD_R + RANDOMX_FREQ_FADD_M + RANDOMX_FREQ_FSUB_R + RANDOMX_FREQ_FSUB_M + RANDOMX_FREQ_FSCAL_R + RANDOMX_FREQ_FMUL_R + RANDOMX_FREQ_FDIV_M + RANDOMX_FREQ_FSQRT_R;

			if (opcode < RANDOMX_FREQ_CBRANCH)
			{
				const uint32_t creg = dst;
#if RANDOMX_PROGRAM_SIZE <= 256
				const uint32_t change = get_byte(registerLastChanged, dst);
				const int32_t lastChanged = (get_byte(registerWasChanged, dst) == 0) ? -1 : static_cast<int32_t>(change);

				// Store condition register and branch target in CBRANCH instruction
				*(uint32_t*)(src_program + i) = (src_inst.x & 0xFF0000FFU) | ((creg | ((lastChanged == -1) ? 0x90 : 0x10)) << 8) | ((static_cast<uint32_t>(lastChanged) & 0xFF) << 16);
#else
				const int32_t lastChanged = registerLastChanged[dst];

				// Store condition register in CBRANCH instruction
				*(uint32_t*)(src_program + i) = (src_inst.x & 0xFF0000FFU) | ((creg | 0x10) << 8);
#endif

				// Mark branch target instruction (src |= 0x40)
				*(uint32_t*)(src_program + lastChanged + 1) |= 0x40 << 8;

#if RANDOMX_PROGRAM_SIZE <= 256
				uint32_t tmp = i | (i << 8);
				registerLastChanged = tmp | (tmp << 16);
				registerLastChanged = registerLastChanged | (registerLastChanged << 32);

				registerWasChanged = 0x0101010101010101ULL;
#else
				registerLastChanged[0] = i;
				registerLastChanged[1] = i;
				registerLastChanged[2] = i;
				registerLastChanged[3] = i;
				registerLastChanged[4] = i;
				registerLastChanged[5] = i;
				registerLastChanged[6] = i;
				registerLastChanged[7] = i;
#endif
			}
		}

		uint64_t registerLatency = 0;
		uint64_t registerReadCycle = 0;
		uint64_t registerLatencyFP = 0;
		uint64_t registerReadCycleFP = 0;
		uint32_t ScratchpadHighLatency = 0;
		uint32_t ScratchpadLatency = 0;

		int32_t first_available_slot = 0;
		int32_t first_allowed_slot_cfround = 0;
		int32_t last_used_slot = -1;
		int32_t last_memory_op_slot = -1;

		uint32_t num_slots_used = 0;
		uint32_t num_instructions = 0;

		int32_t first_instruction_slot = -1;
		bool first_instruction_fp = false;

		//if (global_index == 0)
		//{
		//	for (int j = 0; j < RANDOMX_PROGRAM_SIZE; ++j)
		//	{
		//		print_inst(src_program[j]);
		//		printf("\n");
		//	}
		//	printf("\n");
		//}

		// Schedule instructions
		bool update_branch_target_mark = false;
		bool first_available_slot_is_branch_target = false;
		for (uint32_t i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
		{
			const uint2 inst = src_program[i];

			uint32_t opcode = inst.x & 0xff;
			uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;
			const uint32_t mod = (inst.x >> 24);

			bool is_branch_target = (inst.x & (0x40 << 8)) != 0;
			if (is_branch_target)
			{
				// If an instruction is a branch target, we can't move it before any previous instructions
				first_available_slot = last_used_slot + 1;

				// Mark this slot as a branch target
				// Whatever instruction takes this slot will receive branch target flag
				first_available_slot_is_branch_target = true;
			}

			const uint32_t dst_latency = get_byte(registerLatency, dst);
			const uint32_t src_latency = get_byte(registerLatency, src);
			const uint32_t reg_read_latency = (dst_latency > src_latency) ? dst_latency : src_latency;
			const uint32_t mem_read_latency = ((dst == src) && ((inst.y & ScratchpadL3Mask64) >= RANDOMX_SCRATCHPAD_L2)) ? ScratchpadHighLatency : ScratchpadLatency;

			uint32_t full_read_latency = mem_read_latency;
			update_max(full_read_latency, reg_read_latency);

			uint32_t latency = 0;
			bool is_memory_op = false;
			bool is_memory_store = false;
			bool is_nop = false;
			bool is_branch = false;
			bool is_swap = false;
			bool is_src_read = true;
			bool is_fp = false;
			bool is_cfround = false;

			do {
				if (opcode < RANDOMX_FREQ_IADD_RS)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IADD_RS;

				if (opcode < RANDOMX_FREQ_IADD_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IADD_M;

				if (opcode < RANDOMX_FREQ_ISUB_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_ISUB_R;

				if (opcode < RANDOMX_FREQ_ISUB_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISUB_M;

				if (opcode < RANDOMX_FREQ_IMUL_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IMUL_R;

				if (opcode < RANDOMX_FREQ_IMUL_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IMUL_M;

				if (opcode < RANDOMX_FREQ_IMULH_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IMULH_R;

				if (opcode < RANDOMX_FREQ_IMULH_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IMULH_M;

				if (opcode < RANDOMX_FREQ_ISMULH_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_ISMULH_R;

				if (opcode < RANDOMX_FREQ_ISMULH_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISMULH_M;

				if (opcode < RANDOMX_FREQ_IMUL_RCP)
				{
					is_src_read = false;
					if (inst.y & (inst.y - 1))
						latency = dst_latency;
					else
						is_nop = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IMUL_RCP;

				if (opcode < RANDOMX_FREQ_INEG_R)
				{
					is_src_read = false;
					latency = dst_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_INEG_R;

				if (opcode < RANDOMX_FREQ_IXOR_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IXOR_R;

				if (opcode < RANDOMX_FREQ_IXOR_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IXOR_M;

				if (opcode < RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

				if (opcode < RANDOMX_FREQ_ISWAP_R)
				{
					is_swap = true;
					if (dst != src)
						latency = reg_read_latency;
					else
						is_nop = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISWAP_R;

				if (opcode < RANDOMX_FREQ_FSWAP_R)
				{
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSWAP_R;

				if (opcode < RANDOMX_FREQ_FADD_R)
				{
					dst %= randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FADD_R;

				if (opcode < RANDOMX_FREQ_FADD_M)
				{
					dst %= randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					update_max(latency, src_latency);
					update_max(latency, ScratchpadLatency);
					is_fp = true;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_FADD_M;

				if (opcode < RANDOMX_FREQ_FSUB_R)
				{
					dst %= randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSUB_R;

				if (opcode < RANDOMX_FREQ_FSUB_M)
				{
					dst %= randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					update_max(latency, src_latency);
					update_max(latency, ScratchpadLatency);
					is_fp = true;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_FSUB_M;

				if (opcode < RANDOMX_FREQ_FSCAL_R)
				{
					dst %= randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSCAL_R;

				if (opcode < RANDOMX_FREQ_FMUL_R)
				{
					dst = (dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FMUL_R;

				if (opcode < RANDOMX_FREQ_FDIV_M)
				{
					dst = (dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					update_max(latency, src_latency);
					update_max(latency, ScratchpadLatency);
					is_fp = true;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_FDIV_M;

				if (opcode < RANDOMX_FREQ_FSQRT_R)
				{
					dst = (dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSQRT_R;

				if (opcode < RANDOMX_FREQ_CBRANCH)
				{
					is_src_read = false;
					is_branch = true;
					latency = dst_latency;

					// We can't move CBRANCH before any previous instructions
					first_available_slot = last_used_slot + 1;
					break;
				}
				opcode -= RANDOMX_FREQ_CBRANCH;

				if (opcode < RANDOMX_FREQ_CFROUND)
				{
					latency = src_latency;
					is_cfround = true;
					break;
				}
				opcode -= RANDOMX_FREQ_CFROUND;

				if (opcode < RANDOMX_FREQ_ISTORE)
				{
					latency = reg_read_latency;
					update_max(latency, (last_memory_op_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_memory_op = true;
					is_memory_store = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISTORE;

				is_nop = true;
			} while (false);

			if (is_nop)
			{
				if (is_branch_target)
				{
					// Mark next non-NOP instruction as the branch target instead of this NOP
					update_branch_target_mark = true;
				}
				continue;
			}

			if (update_branch_target_mark)
			{
				*(uint32_t*)(src_program + i) |= 0x40 << 8;
				update_branch_target_mark = false;
				is_branch_target = true;
			}

			int32_t first_allowed_slot = first_available_slot;
			update_max(first_allowed_slot, latency * WORKERS_PER_HASH);
			if (is_cfround)
				update_max(first_allowed_slot, first_allowed_slot_cfround);
			else
				update_max(first_allowed_slot, get_byte(is_fp ? registerReadCycleFP : registerReadCycle, dst) * WORKERS_PER_HASH);

			if (is_swap)
				update_max(first_allowed_slot, get_byte(registerReadCycle, src) * WORKERS_PER_HASH);

			int32_t slot_to_use = last_used_slot + 1;
			update_max(slot_to_use, first_allowed_slot);

			if (is_fp)
			{
				slot_to_use = -1;
				for (int32_t j = first_allowed_slot; slot_to_use < 0; ++j)
				{
					if ((execution_plan[j] == 0) && (execution_plan[j + 1] == 0) && ((j + 1) % WORKERS_PER_HASH))
					{
						bool blocked = false;
						for (int32_t k = (j / WORKERS_PER_HASH) * WORKERS_PER_HASH; k < j; ++k)
						{
							if (execution_plan[k] || (k == first_instruction_slot))
							{
								const uint32_t inst = src_program[execution_plan[k]].x;

								// If there is an integer instruction which is a branch target or a branch, or this FP instruction is a branch target itself, we can't reorder it to add more FP instructions to this cycle
								if (((inst & (0x20 << 8)) == 0) && (((inst & (0x50 << 8)) != 0) || is_branch_target))
								{
									blocked = true;
									continue;
								}
							}
						}

						if (!blocked)
						{
							for (int32_t k = (j / WORKERS_PER_HASH) * WORKERS_PER_HASH; k < j; ++k)
							{
								if (execution_plan[k] || (k == first_instruction_slot))
								{
									const uint32_t inst = src_program[execution_plan[k]].x;
									if ((inst & (0x20 << 8)) == 0)
									{
										execution_plan[j] = execution_plan[k];
										execution_plan[j + 1] = execution_plan[k + 1];
										if (first_instruction_slot == k) first_instruction_slot = j;
										if (first_instruction_slot == k + 1) first_instruction_slot = j + 1;
										slot_to_use = k;
										break;
									}
								}
							}

							if (slot_to_use < 0)
							{
								slot_to_use = j;
							}

							break;
						}
					}
				}
			}
			else
			{
				for (int32_t j = first_allowed_slot; j <= last_used_slot; ++j)
				{
					if (execution_plan[j] == 0)
					{
						slot_to_use = j;
						break;
					}
				}
			}

			if (i == 0)
			{
				first_instruction_slot = slot_to_use;
				first_instruction_fp = is_fp;
			}

			if (is_cfround)
			{
				first_allowed_slot_cfround = slot_to_use - (slot_to_use % WORKERS_PER_HASH) + WORKERS_PER_HASH;
			}

			++num_instructions;

			execution_plan[slot_to_use] = i;
			++num_slots_used;

			if (is_fp)
			{
				execution_plan[slot_to_use + 1] = i;
				++num_slots_used;
			}

			const uint32_t next_latency = (slot_to_use / WORKERS_PER_HASH) + 1;

			if (is_src_read)
			{
				int32_t value = get_byte(registerReadCycle, src);
				update_max(value, slot_to_use / WORKERS_PER_HASH);
				set_byte(registerReadCycle, src, value);
			}

			if (is_memory_op)
			{
				update_max(last_memory_op_slot, slot_to_use);
			}

			if (is_cfround)
			{
				const uint32_t t = next_latency | (next_latency << 8);
				registerLatencyFP = t | (t << 16);
				registerLatencyFP = registerLatencyFP | (registerLatencyFP << 32);
			}
			else if (is_fp)
			{
				set_byte(registerLatencyFP, dst, next_latency);

				int32_t value = get_byte(registerReadCycleFP, dst);
				update_max(value, slot_to_use / WORKERS_PER_HASH);
				set_byte(registerReadCycleFP, dst, value);
			}
			else
			{
				if (!is_memory_store && !is_nop)
				{
					set_byte(registerLatency, dst, next_latency);
					if (is_swap)
						set_byte(registerLatency, src, next_latency);

					int32_t value = get_byte(registerReadCycle, dst);
					update_max(value, slot_to_use / WORKERS_PER_HASH);
					set_byte(registerReadCycle, dst, value);
				}

				if (is_branch)
				{
					const uint32_t t = next_latency | (next_latency << 8);
					registerLatency = t | (t << 16);
					registerLatency = registerLatency | (registerLatency << 32);
				}

				if (is_memory_store)
				{
					int32_t value = get_byte(registerReadCycle, dst);
					update_max(value, slot_to_use / WORKERS_PER_HASH);
					set_byte(registerReadCycle, dst, value);
					ScratchpadLatency = slot_to_use / WORKERS_PER_HASH;
					if ((mod >> 4) >= randomx::StoreL3Condition)
						ScratchpadHighLatency = slot_to_use / WORKERS_PER_HASH;
				}
			}

			if (execution_plan[first_available_slot] || (first_available_slot == first_instruction_slot))
			{
				if (first_available_slot_is_branch_target)
				{
					src_program[i].x |= 0x40 << 8;
					first_available_slot_is_branch_target = false;
				}

				if (is_fp)
					++first_available_slot;

				do {
					++first_available_slot;
				} while ((first_available_slot < RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH) && (execution_plan[first_available_slot] != 0));
			}

			if (is_branch_target)
			{
				update_max(first_available_slot, is_fp ? (slot_to_use + 2) : (slot_to_use + 1));
			}

			update_max(last_used_slot, is_fp ? (slot_to_use + 1) : slot_to_use);
			while (execution_plan[last_used_slot] || (last_used_slot == first_instruction_slot) || ((last_used_slot == first_instruction_slot + 1) && first_instruction_fp))
			{
				++last_used_slot;
			}
			--last_used_slot;

			if (is_fp && (last_used_slot >= first_allowed_slot_cfround))
				first_allowed_slot_cfround = last_used_slot + 1;

			//if (global_index == 0)
			//{
			//	printf("slot_to_use = %d, first_available_slot = %d, last_used_slot = %d\n", slot_to_use, first_available_slot, last_used_slot);
			//	for (int j = 0; j <= last_used_slot; ++j)
			//	{
			//		if (execution_plan[j] || (j == first_instruction_slot) || ((j == first_instruction_slot + 1) && first_instruction_fp))
			//		{
			//			print_inst(src_program[execution_plan[j]]);
			//			printf(" | ");
			//		}
			//		else
			//		{
			//			printf("                      | ");
			//		}
			//		if (((j + 1) % WORKERS_PER_HASH) == 0) printf("\n");
			//	}
			//	printf("\n\n");
			//}
		}

		//if (global_index == 0)
		//{
		//	printf("IPC = %.3f, WPC = %.3f, num_instructions = %u, num_slots_used = %u, first_instruction_slot = %d, last_used_slot = %d, registerLatency = %016llx, registerLatencyFP = %016llx \n",
		//		num_instructions / static_cast<double>(last_used_slot / WORKERS_PER_HASH + 1),
		//		num_slots_used / static_cast<double>(last_used_slot / WORKERS_PER_HASH + 1),
		//		num_instructions,
		//		num_slots_used,
		//		first_instruction_slot,
		//		last_used_slot,
		//		registerLatency,
		//		registerLatencyFP
		//	);

		//	//for (int j = 0; j < RANDOMX_PROGRAM_SIZE; ++j)
		//	//{
		//	//	print_inst(src_program[j]);
		//	//	printf("\n");
		//	//}
		//	//printf("\n");

		//	for (int j = 0; j <= last_used_slot; ++j)
		//	{
		//		if (execution_plan[j] || (j == first_instruction_slot) || ((j == first_instruction_slot + 1) && first_instruction_fp))
		//		{
		//			print_inst(src_program[execution_plan[j]]);
		//			printf(" | ");
		//		}
		//		else
		//		{
		//			printf("                      | ");
		//		}
		//		if (((j + 1) % WORKERS_PER_HASH) == 0) printf("\n");
		//	}
		//	printf("\n\n");
		//}

		uint32_t ma = static_cast<uint32_t>(entropy[8]) & CacheLineAlignMask;
		uint32_t mx = static_cast<uint32_t>(entropy[10]) & CacheLineAlignMask;

		uint32_t addressRegisters = static_cast<uint32_t>(entropy[12]);
		addressRegisters = ((addressRegisters & 1) | (((addressRegisters & 2) ? 3U : 2U) << 8) | (((addressRegisters & 4) ? 5U : 4U) << 16) | (((addressRegisters & 8) ? 7U : 6U) << 24)) * sizeof(uint64_t);

		uint32_t datasetOffset = (entropy[13] & randomx::DatasetExtraItems) * randomx::CacheLineSize;

		ulonglong2 eMask = *(ulonglong2*)(entropy + 14);
		eMask.x = getFloatMask(eMask.x);
		eMask.y = getFloatMask(eMask.y);

		((uint32_t*)(R + 16))[0] = ma;
		((uint32_t*)(R + 16))[1] = mx;
		((uint32_t*)(R + 16))[2] = addressRegisters;
		((uint32_t*)(R + 16))[3] = datasetOffset;
		((ulonglong2*)(R + 18))[0] = eMask;

		uint32_t* imm_buf = (uint32_t*)(R + REGISTERS_SIZE / sizeof(uint64_t));
		uint32_t imm_index = 0;
		int32_t imm_index_fscal_r = -1;
		uint32_t* compiled_program = (uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t));

		// Generate opcodes for execute_vm
		int32_t branch_target_slot = -1;
		int32_t k = -1;
		for (int32_t i = 0; i <= last_used_slot; ++i)
		{
			if (!(execution_plan[i] || (i == first_instruction_slot) || ((i == first_instruction_slot + 1) && first_instruction_fp)))
				continue;

			uint32_t num_workers = 1;
			uint32_t num_fp_insts = 0;
			while ((i + num_workers <= last_used_slot) && ((i + num_workers) % WORKERS_PER_HASH) && (execution_plan[i + num_workers] || (i + num_workers == first_instruction_slot) || ((i + num_workers == first_instruction_slot + 1) && first_instruction_fp)))
			{
				if ((num_workers & 1) && ((src_program[execution_plan[i + num_workers]].x & (0x20 << 8)) != 0))
					++num_fp_insts;
				++num_workers;
			}

			//if (global_index == 0)
			//	printf("i = %d, num_workers = %u, num_fp_insts = %u\n", i, num_workers, num_fp_insts);

			num_workers = ((num_workers - 1) << NUM_INSTS_OFFSET) | (num_fp_insts << NUM_FP_INSTS_OFFSET);

			const uint2 src_inst = src_program[execution_plan[i]];
			uint2 inst = src_inst;

			uint32_t opcode = inst.x & 0xff;
			const uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;
			const uint32_t mod = (inst.x >> 24);

			const bool is_fp = (src_inst.x & (0x20 << 8)) != 0;
			if (is_fp && ((i & 1) == 0))
				++i;

			const bool is_branch_target = (src_inst.x & (0x40 << 8)) != 0;
			if (is_branch_target && (branch_target_slot < 0))
				branch_target_slot = k;

			++k;

			inst.x = INST_NOP;

			if (opcode < RANDOMX_FREQ_IADD_RS)
			{
				const uint32_t shift = (mod >> 2) % 4;

				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (shift << SHIFT_OFFSET);

				if (dst != randomx::RegisterNeedsDisplacement)
				{
					// Encode regular ADD (opcode 1)
					inst.x |= (1 << OPCODE_OFFSET);
				}
				else
				{
					// Encode ADD with src and imm32 (opcode 0)
					inst.x |= imm_index << IMM_OFFSET;
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_RS;

			if (opcode < RANDOMX_FREQ_IADD_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (1 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_M;

			if (opcode < RANDOMX_FREQ_ISUB_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISUB_R;

			if (opcode < RANDOMX_FREQ_ISUB_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (1 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISUB_M;

			if (opcode < RANDOMX_FREQ_IMUL_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (2 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_R;

			if (opcode < RANDOMX_FREQ_IMUL_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (2 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_M;

			if (opcode < RANDOMX_FREQ_IMULH_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (6 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMULH_R;

			if (opcode < RANDOMX_FREQ_IMULH_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (6 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMULH_M;

			if (opcode < RANDOMX_FREQ_ISMULH_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (4 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISMULH_R;

			if (opcode < RANDOMX_FREQ_ISMULH_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (4 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISMULH_M;

			if (opcode < RANDOMX_FREQ_IMUL_RCP)
			{
				const uint64_t r = imul_rcp_value(inst.y);
				if (r == 1)
				{
					*(compiled_program++) = INST_NOP | num_workers;
					continue;
				}

				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (2 << OPCODE_OFFSET);
				inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM64_OFFSET);

				if (imm_index < IMM_INDEX_COUNT - 1)
				{
					imm_buf[imm_index] = ((const uint32_t*)&r)[0];
					imm_buf[imm_index + 1] = ((const uint32_t*)&r)[1];
					imm_index += 2;
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_RCP;

			if (opcode < RANDOMX_FREQ_INEG_R)
			{
				inst.x = (dst << DST_OFFSET) | (5 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_INEG_R;

			if (opcode < RANDOMX_FREQ_IXOR_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (3 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IXOR_R;

			if (opcode < RANDOMX_FREQ_IXOR_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (3 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IXOR_M;

			if (opcode < RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (7 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}
				if (opcode >= RANDOMX_FREQ_IROR_R)
				{
					inst.x |= (1 << NEGATIVE_SRC_OFFSET);
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

			if (opcode < RANDOMX_FREQ_ISWAP_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (8 << OPCODE_OFFSET);

				*(compiled_program++) = ((src != dst) ? inst.x : INST_NOP) | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISWAP_R;

			if (opcode < RANDOMX_FREQ_FSWAP_R)
			{
				inst.x = (dst << DST_OFFSET) | (11 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSWAP_R;

			if (opcode < RANDOMX_FREQ_FADD_R)
			{
				inst.x = ((dst % randomx::RegisterCountFlt) << DST_OFFSET) | ((src % randomx::RegisterCountFlt) << (SRC_OFFSET + 1)) | (12 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FADD_R;

			if (opcode < RANDOMX_FREQ_FADD_M)
			{
				const uint32_t location = (mod % 4) ? 1 : 2;
				inst.x = ((dst % randomx::RegisterCountFlt) << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (12 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FADD_M;

			if (opcode < RANDOMX_FREQ_FSUB_R)
			{
				inst.x = ((dst % randomx::RegisterCountFlt) << DST_OFFSET) | ((src % randomx::RegisterCountFlt) << (SRC_OFFSET + 1)) | (12 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSUB_R;

			if (opcode < RANDOMX_FREQ_FSUB_M)
			{
				const uint32_t location = (mod % 4) ? 1 : 2;
				inst.x = ((dst % randomx::RegisterCountFlt) << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (12 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSUB_M;

			if (opcode < RANDOMX_FREQ_FSCAL_R)
			{
				inst.x = ((dst % randomx::RegisterCountFlt) << DST_OFFSET) | (1 << SRC_IS_IMM64_OFFSET) | (3 << OPCODE_OFFSET);
				if (imm_index_fscal_r >= 0)
				{
					inst.x |= (imm_index_fscal_r << IMM_OFFSET);
				}
				else
				{
					imm_index_fscal_r = imm_index;
					inst.x |= (imm_index << IMM_OFFSET);

					if (imm_index < IMM_INDEX_COUNT - 1)
					{
						imm_buf[imm_index] = 0;
						imm_buf[imm_index + 1] = 0x80F00000UL;
						imm_index += 2;
					}
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSCAL_R;

			if (opcode < RANDOMX_FREQ_FMUL_R)
			{
				inst.x = (((dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt) << DST_OFFSET) | ((src % randomx::RegisterCountFlt) << (SRC_OFFSET + 1)) | (1 << SHIFT_OFFSET) | (12 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FMUL_R;

			if (opcode < RANDOMX_FREQ_FDIV_M)
			{
				const uint32_t location = (mod % 4) ? 1 : 2;
				inst.x = (((dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt) << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (15 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FDIV_M;

			if (opcode < RANDOMX_FREQ_FSQRT_R)
			{
				inst.x = (((dst % randomx::RegisterCountFlt) + randomx::RegisterCountFlt) << DST_OFFSET) | (14 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSQRT_R;

			if (opcode < RANDOMX_FREQ_CBRANCH)
			{
				inst.x = (dst << DST_OFFSET) | (9 << OPCODE_OFFSET);
				inst.x |= (imm_index << IMM_OFFSET);

				const uint32_t cshift = (mod >> 4) + randomx::ConditionOffset;

				uint32_t imm = inst.y | (1U << cshift);
				if (cshift > 0)
					imm &= ~(1U << (cshift - 1));

				if (imm_index < IMM_INDEX_COUNT - 1)
				{
					imm_buf[imm_index] = imm;
					imm_buf[imm_index + 1] = cshift | (static_cast<uint32_t>(branch_target_slot) << 5);
					imm_index += 2;
				}
				else
				{
					// Data doesn't fit, skip it
					inst.x = INST_NOP;
				}

				branch_target_slot = -1;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_CBRANCH;

			if (opcode < RANDOMX_FREQ_CFROUND)
			{
				inst.x = (src << SRC_OFFSET) | (13 << OPCODE_OFFSET) | ((inst.y & 63) << IMM_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_CFROUND;

			if (opcode < RANDOMX_FREQ_ISTORE)
			{
				const uint32_t location = ((mod >> 4) >= randomx::StoreL3Condition) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (10 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;
				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISTORE;

			*(compiled_program++) = inst.x | num_workers;
		}

		((uint32_t*)(R + 20))[0] = static_cast<uint32_t>(compiled_program - (uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t)));
	}
}

template<typename T, size_t N>
__device__ void load_buffer(T (&dst_buf)[N], const void* src_buf)
{
	uint32_t i = threadIdx.x * sizeof(T);
	const uint32_t step = blockDim.x * sizeof(T);
	const uint8_t* src = ((const uint8_t*) src_buf) + blockIdx.x * sizeof(T) * N + i;
	uint8_t* dst = ((uint8_t*) dst_buf) + i;
	while (i < sizeof(T) * N)
	{
		*(T*)(dst) = *(T*)(src);
		src += step;
		dst += step;
		i += step;
	}
}

template<int> __device__ double fma_rnd(double a, double b, double c, uint32_t fprc);
template<int, bool> __device__ double div_rnd(double a, double b, uint32_t fprc);
template<int, bool> __device__ double sqrt_rnd(double x, uint32_t fprc);

template<> __device__ double fma_rnd<0>(double a, double b, double c, uint32_t) { return __fma_rn(a, b, c); }
template<> __device__ double fma_rnd<1>(double a, double b, double c, uint32_t) { return __fma_rd(a, b, c); }
template<> __device__ double fma_rnd<2>(double a, double b, double c, uint32_t) { return __fma_ru(a, b, c); }
template<> __device__ double fma_rnd<3>(double a, double b, double c, uint32_t) { return __fma_rz(a, b, c); }

template<> __device__ double div_rnd<0, false>(double a, double b, uint32_t) { return __ddiv_rn(a, b); }
template<> __device__ double div_rnd<0, true>(double a, double b, uint32_t) { return __ddiv_rn(a, b); }
template<> __device__ double div_rnd<1, true>(double a, double b, uint32_t) { return __ddiv_rd(a, b); }
template<> __device__ double div_rnd<2, true>(double a, double b, uint32_t) { return __ddiv_ru(a, b); }

template<> __device__ double sqrt_rnd<0, false>(double a, uint32_t) { return __dsqrt_rn(a); }
template<> __device__ double sqrt_rnd<0, true>(double a, uint32_t) { return __dsqrt_rn(a); }
template<> __device__ double sqrt_rnd<1, true>(double a, uint32_t) { return __dsqrt_rd(a); }
template<> __device__ double sqrt_rnd<2, true>(double a, uint32_t) { return __dsqrt_ru(a); }

template<> __device__ double fma_rnd<-1>(double a, double b, double c, uint32_t fprc)
{
	if (fprc == 0)
		return fma_rnd<0>(a, b, c, 0);
	else if (fprc == 1)
		return fma_rnd<1>(a, b, c, 1);
	else if (fprc == 2)
		return fma_rnd<2>(a, b, c, 2);
	else
		return fma_rnd<3>(a, b, c, 3);
}

template<> __device__ double div_rnd<-1, true>(double a, double b, uint32_t fprc)
{
	if (fprc == 0)
		return div_rnd<0, true>(a, b, 0);
	else if (fprc == 2)
		return div_rnd<2, true>(a, b, 2);
	else
		return div_rnd<1, true>(a, b, 1);
}

template<> __device__ double sqrt_rnd<-1, true>(double a, uint32_t fprc)
{
	if (fprc == 0)
		return sqrt_rnd<0, true>(a, 0);
	else if (fprc == 2)
		return sqrt_rnd<2, true>(a, 2);
	else
		return sqrt_rnd<1, true>(a, 1);
}

template<> __device__ double div_rnd<-1, false>(double a, double b, uint32_t fprc)
{
	// Initial approximation
	double y0;
	asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(y0) : "d"(b));

	// Improve initial approximation (can be skipped)
	// 1 of 2^31 quotients will be incorrect in the last bit without it (1 incorrect hash per ~32768 hashes)
	//y0 = __fma_rn(y0, __fma_rn(-b, y0, 1.0), y0);

	// First Newton-Raphson iteration
	const double y1 = __fma_rn(y0, __fma_rn(-b, y0, 1.0), y0);
	const double t0 = a * y1;
	const double t1 = __fma_rn(-b, t0, a);

	// Second Newton-Raphson iteration
	double result = __fma_rn(y1, t1, t0);

	// Get result in the other 2 rounding modes
	if (fprc)
	{
		const double result_rd = __fma_rd(y1, t1, t0);
		const double result_ru = __fma_ru(y1, t1, t0);
		if (fprc & 1) result = result_rd;
		if (fprc == 2) result = result_ru;
	}

	// Check for infinity/NaN
	const uint64_t inf = 2047ULL << 52;
	const uint64_t inf_rnd = inf - (fprc & 1);

	if (((*((uint64_t*) &result) >> 52) & 2047) == 2047) result = *((const double*) &inf_rnd);
	if (*((uint64_t*) &a) == inf) result = a;

	// Check for exact equality
	if (a == b) result = 1.0;

	//double check;
	//if (fprc == 0)
	//	check = __ddiv_rn(a, b);
	//else if (fprc == 2)
	//	check = __ddiv_ru(a, b);
	//else
	//	check = __ddiv_rd(a, b);

	//if (result != check)
	//{
	//	const uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
	//	const int64_t delta = *(int64_t*)(&result) - *(int64_t*)(&check);
	//	printf("idx = %u fprc = %u a = %016llx b = %016llx result = %016llx check = %016llx d = %lld\n", idx, fprc, a, b, result, check, delta);
	//}

	return result;
}

template<> __device__ double sqrt_rnd<-1, false>(double x, uint32_t fprc)
{
	// Initial approximation
	double y0, t0, t1;
	asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(y0) : "d"(x));

	// Improve initial approximation (can be skipped)
	// 1 of 2^28 square roots will be incorrect in the last bit without it (1 incorrect hash per ~2731 hashes)
	//y0 = __fma_rn(y0, __fma_rn(y0 * -0.5, y0 * x, 0.5), y0);

	// First Newton-Raphson iteration
	t0 = y0 * x;
	t1 = y0 * -0.5;
	t1 = __fma_rn(t1, t0, 0.5);					// 0.5 * (1.0 - y0 * y0 * x)
	const double y1_x = __fma_rn(t0, t1, t0);	// y1 * x = 0.5 * y0 * x * (3.0 - y0 * y0 * x)

	// Second Newton-Raphson iteration
	y0 *= 0.5;
	y0 = __fma_rn(y0, t1, y0);					// 0.5 * y1
	t1 = __fma_rn(-y1_x, y1_x, x);				// x * (1.0 - x * y1 * y1)

	double result;
	result = __fma_rn(t1, y0, y1_x);		// x * 0.5 * y1 * (3.0 - x * y1 * y1)

	// Get result in the other 2 rounding modes
	if (fprc)
	{
		const double result_rd = __fma_rd(t1, y0, y1_x);
		const double result_ru = __fma_ru(t1, y0, y1_x);
		if (fprc & 1) result = result_rd;
		if (fprc == 2) result = result_ru;
	}

	// Check for infinity
	if (*((uint64_t*)&x) == (2047ULL << 52)) result = x;

	//double check;
	//if (fprc == 0)
	//	check = __dsqrt_rn(x);
	//else if (fprc == 2)
	//	check = __dsqrt_ru(x);
	//else
	//	check = __dsqrt_rd(x);

	//if (result != check)
	//{
	//	const uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
	//	const int64_t delta = *(int64_t*)(&result) - *(int64_t*)(&check);
	//	printf("idx = %u x = %e d = %lld\n", idx, x, delta);
	//}

	return result;
}

template<int WORKERS_PER_HASH, int ROUNDING_MODE, bool HIGH_PRECISION>
__device__ void inner_loop(
	const uint32_t program_length,
	const uint32_t* compiled_program,
	const int32_t sub,
	uint8_t* scratchpad,
	const uint32_t fp_reg_offset,
	const uint32_t fp_reg_group_A_offset,
	uint64_t* R,
	uint32_t* imm_buf,
	const uint32_t batch_size,
	uint32_t &fprc,
	const uint64_t xexponentMask,
	const uint32_t workers_mask
)
{
	const int32_t sub2 = sub >> 1;
	imm_buf[IMM_INDEX_COUNT + 1] = fprc;

	#pragma unroll(1)
	for (int32_t ip = 0; ip < program_length;)
	{
		imm_buf[IMM_INDEX_COUNT] = ip;

		uint32_t inst = compiled_program[ip];
		const int32_t num_workers = (inst >> NUM_INSTS_OFFSET) & (WORKERS_PER_HASH - 1);
		const int32_t num_fp_insts = (inst >> NUM_FP_INSTS_OFFSET) & (WORKERS_PER_HASH - 1);
		const int32_t num_insts = num_workers - num_fp_insts;

		if (sub <= num_workers)
		{
			const int32_t inst_offset = sub - num_fp_insts;
			const bool is_fp = inst_offset < num_fp_insts;
			inst = compiled_program[ip + (is_fp ? sub2 : inst_offset)];
			//if ((idx == 0) && (ic == 0))
			//{
			//	printf("num_fp_insts = %u, sub = %u, ip = %u, inst = %08x\n", num_fp_insts, sub, ip + ((sub < num_fp_insts * 2) ? (sub / 2) : (sub - num_fp_insts)), inst);
			//}

			asm("// INSTRUCTION DECODING BEGIN");

			uint32_t opcode = (inst >> OPCODE_OFFSET) & 15;
			const uint32_t location = (inst >> LOC_OFFSET) & 1;

			const uint32_t reg_size_shift = is_fp ? 4 : 3;
			const uint32_t reg_base_offset = is_fp ? fp_reg_offset : 0;
			const uint32_t reg_base_src_offset = is_fp ? fp_reg_group_A_offset : 0;

			uint32_t dst_offset = (inst >> DST_OFFSET) & 7;
			dst_offset = reg_base_offset + (dst_offset << reg_size_shift);

			uint32_t src_offset = (inst >> SRC_OFFSET) & 7;
			src_offset = (src_offset << 3) + (location ? 0 : reg_base_src_offset);

			uint64_t* dst_ptr = (uint64_t*)((uint8_t*)(R) + dst_offset);
			uint64_t* src_ptr = (uint64_t*)((uint8_t*)(R) + src_offset);

			const uint32_t imm_offset = (inst >> IMM_OFFSET) & 255;
			const uint32_t* imm_ptr = imm_buf + imm_offset;

			uint64_t dst = *dst_ptr;
			uint64_t src = *src_ptr;
			uint2 imm;
			imm.x = imm_ptr[0];
			imm.y = imm_ptr[1];

			asm("// INSTRUCTION DECODING END");

			if (location)
			{
				asm("// SCRATCHPAD ACCESS BEGIN");

				uint32_t loc_shift;
				asm("bfe.u32 %0, %1, 21, 5;" : "=r"(loc_shift) : "r"(imm.x));
				const uint32_t mask = (0xFFFFFFFFU >> loc_shift) - 7;

				const bool is_read = (opcode != 10);
				uint32_t addr = is_read ? ((loc_shift == LOC_L3) ? 0 : static_cast<uint32_t>(src)) : static_cast<uint32_t>(dst);
				addr += static_cast<int32_t>(imm.x);
				addr &= mask;

				uint64_t* ptr = (uint64_t*)(scratchpad + addr);

				if (is_read)
				{
					src = *ptr;
				}
				else
				{
					*ptr = src;
					goto execution_end;
				}

				asm("// SCRATCHPAD ACCESS END");
			}

			{
				asm("// EXECUTION BEGIN");

				if (inst & (1 << SRC_IS_IMM32_OFFSET)) src = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(imm.x)));

				// Check instruction opcodes (most frequent instructions come first)
				if (opcode <= 3)
				{
					asm("// IADD_RS, IADD_M, ISUB_R, ISUB_M, IMUL_R, IMUL_M, IMUL_RCP, IXOR_R, IXOR_M, FSCAL_R (109/256) ------>");
					if (inst & (1 << NEGATIVE_SRC_OFFSET)) src = static_cast<uint64_t>(-static_cast<int64_t>(src));
					if (opcode == 0) dst += static_cast<int32_t>(imm.x);
					const uint32_t shift = (inst >> SHIFT_OFFSET) & 3;
					if (opcode < 2) dst += src << shift;
					const uint64_t imm64 = *((uint64_t*) &imm);
					if (inst & (1 << SRC_IS_IMM64_OFFSET)) src = imm64;
					if (opcode == 2) dst *= src;
					if (opcode == 3) dst ^= src;
					asm("// <------ IADD_RS, IADD_M, ISUB_R, ISUB_M, IMUL_R, IMUL_M, IMUL_RCP, IXOR_R, IXOR_M, FSCAL_R (109/256)");
				}
				else if (opcode == 12)
				{
					asm("// FADD_R, FADD_M, FSUB_R, FSUB_M, FMUL_R (74/256) ------>");

					if (location) src = bit_cast<uint64_t>(__int2double_rn(static_cast<int32_t>(src >> ((sub & 1) * 32))));
					if (inst & (1 << NEGATIVE_SRC_OFFSET)) src ^= 0x8000000000000000ULL;

					const bool is_mul = (inst & (1 << SHIFT_OFFSET)) != 0;
					const double a = __longlong_as_double(dst);
					const double b = __longlong_as_double(src);

					dst = bit_cast<uint64_t>(fma_rnd<ROUNDING_MODE>(a, is_mul ? b : 1.0, is_mul ? 0.0 : b, fprc));

					asm("// <------ FADD_R, FADD_M, FSUB_R, FSUB_M, FMUL_R (74/256)");
				}
				else if (opcode == 9)
				{
					asm("// CBRANCH (16/256) ------>");
					dst += static_cast<int32_t>(imm.x);
					if ((static_cast<uint32_t>(dst) & (randomx::ConditionMask << (imm.y & 31))) == 0)
					{
						imm_buf[IMM_INDEX_COUNT] = static_cast<uint32_t>((static_cast<int32_t>(imm.y) >> 5) - num_insts);
					}
					asm("// <------ CBRANCH (16/256)");
				}
				else if (opcode == 7)
				{
					asm("// IROR_R, IROL_R (10/256) ------>");
					uint32_t shift1 = src & 63;
#if RANDOMX_FREQ_IROL_R > 0
					const uint32_t shift2 = 64 - shift1;
					const bool is_rol = (inst & (1 << NEGATIVE_SRC_OFFSET));
					dst = (dst >> (is_rol ? shift2 : shift1)) | (dst << (is_rol ? shift1 : shift2));
#else
					dst = (dst >> shift1) | (dst << (64 - shift1));
#endif
					asm("// <------ IROR_R, IROL_R (10/256)");
				}
				else if (opcode == 14)
				{
					asm("// FSQRT_R (6/256) ------>");
					dst = bit_cast<uint64_t>(sqrt_rnd<ROUNDING_MODE, HIGH_PRECISION>(__longlong_as_double(dst), fprc));
					asm("// <------ FSQRT_R (6/256)");
				}
				else if (opcode == 6)
				{
					asm("// IMULH_R, IMULH_M (5/256) ------>");
					dst = __umul64hi(dst, src);
					asm("// <------ IMULH_R, IMULH_M (5/256)");
				}
				else if (opcode == 4)
				{
					asm("// ISMULH_R, ISMULH_M (5/256) ------>");
					dst = static_cast<uint64_t>(__mul64hi(static_cast<int64_t>(dst), static_cast<int64_t>(src)));
					asm("// <------ ISMULH_R, ISMULH_M (5/256)");
				}
				else if (opcode == 11)
				{
					asm("// FSWAP_R (4/256) ------>");
					dst = *(uint64_t*)((uint8_t*)(R) + (dst_offset ^ 8));
					asm("// <------ FSWAP_R (4/256)");
				}
				else if (opcode == 8)
				{
					asm("// ISWAP_R (4/256) ------>");
					*src_ptr = dst;
					dst = src;
					asm("// <------ ISWAP_R (4/256)");
				}
				else if (opcode == 15)
				{
					asm("// FDIV_M (4/256) ------>");
					src = bit_cast<uint64_t>(__int2double_rn(static_cast<int32_t>(src >> ((sub & 1) * 32))));
					src &= randomx::dynamicMantissaMask;
					src |= xexponentMask;
					dst = bit_cast<uint64_t>(div_rnd<ROUNDING_MODE, HIGH_PRECISION>(__longlong_as_double(dst), __longlong_as_double(src), fprc));
					asm("// <------ FDIV_M (4/256)");
				}
				else if (opcode == 5)
				{
					asm("// INEG_R (2/256) ------>");
					dst = static_cast<uint64_t>(-static_cast<int64_t>(dst));
					asm("// <------ INEG_R (2/256)");
				}
				// CFROUND check will be skipped and removed entirely by the compiler if ROUNDING_MODE >= 0
				else if (ROUNDING_MODE < 0)
				{
					asm("// CFROUND (1/256) ------>");
					imm_buf[IMM_INDEX_COUNT + 1] = ((src >> imm_offset) | (src << (64 - imm_offset))) & 3;
					asm("// <------ CFROUND (1/256)");
					goto execution_end;
				}

				*dst_ptr = dst;
				asm("// EXECUTION END");
			}
		}

		execution_end:
		{
			asm("// SYNCHRONIZATION OF INSTRUCTION POINTER AND ROUNDING MODE BEGIN");

#			if ((__CUDA_ARCH__ >= 300) && (__CUDACC_VER_MAJOR__ >= 9))
			asm("bar.warp.sync %0;" :: "r"(workers_mask));
#			endif

			ip = imm_buf[IMM_INDEX_COUNT];
			fprc = imm_buf[IMM_INDEX_COUNT + 1];

			asm("// SYNCHRONIZATION OF INSTRUCTION POINTER AND ROUNDING MODE END");

			ip += num_insts + 1;
		}
	}
}

template<int WORKERS_PER_HASH, bool HIGH_PRECISION>
__global__ void __launch_bounds__((WORKERS_PER_HASH == 16) ? 32 : 16, 16) execute_vm(void* vm_states, void* rounding, void* scratchpads, const void* dataset_ptr, uint32_t batch_size, uint32_t num_iterations, bool first, bool last)
{
	// 2 hashes per warp, 4 KB shared memory for VM states
	__shared__ uint64_t vm_states_local[(VM_STATE_SIZE * 2) / sizeof(uint64_t)];

	load_buffer(vm_states_local, vm_states);

#	if (__CUDACC_VER_MAJOR__ >= 9)
	__syncwarp();
#	else
	__syncthreads();
#	endif

	enum { IDX_WIDTH = (WORKERS_PER_HASH == 16) ? 16 : 8 };

	uint64_t* R = vm_states_local + (threadIdx.x / IDX_WIDTH) * VM_STATE_SIZE / sizeof(uint64_t);
	double* F = (double*)(R + 8);
	double* E = (double*)(R + 16);

	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int32_t idx = global_index / IDX_WIDTH;
	const int32_t sub = global_index % IDX_WIDTH;

	uint32_t ma = ((uint32_t*)(R + 16))[0];
	uint32_t mx = ((uint32_t*)(R + 16))[1];

	const uint32_t addressRegisters = ((uint32_t*)(R + 16))[2];
	const uint64_t* readReg0 = (uint64_t*)(((uint8_t*) R) + (addressRegisters & 0xff));
	const uint64_t* readReg1 = (uint64_t*)(((uint8_t*) R) + ((addressRegisters >> 8) & 0xff));
	const uint32_t* readReg2 = (uint32_t*)(((uint8_t*) R) + ((addressRegisters >> 16) & 0xff));
	const uint32_t* readReg3 = (uint32_t*)(((uint8_t*) R) + (addressRegisters >> 24));

	const uint32_t datasetOffset = ((uint32_t*)(R + 16))[3];
	const uint8_t* dataset = ((const uint8_t*) dataset_ptr) + datasetOffset;

	const uint32_t fp_reg_offset = 64 + ((global_index & 1) << 3);
	const uint32_t fp_reg_group_A_offset = 192 + ((global_index & 1) << 3);

	ulonglong2 eMask = ((ulonglong2*)(R + 18))[0];

	const uint32_t program_length = ((uint32_t*)(R + 20))[0];
	uint32_t fprc = ((uint32_t*) rounding)[idx];

	uint32_t spAddr0 = first ? mx : 0;
	uint32_t spAddr1 = first ? ma : 0;

	uint8_t* scratchpad = ((uint8_t*) scratchpads) + idx * static_cast<uint64_t>(RANDOMX_SCRATCHPAD_L3 + 64);

	const bool f_group = (sub < 4);

	double* fe = f_group ? (F + sub * 2) : (E + (sub - 4) * 2);
	double* f = F + sub;
	double* e = E + sub;

	const uint64_t andMask = f_group ? uint64_t(-1) : randomx::dynamicMantissaMask;
	const uint64_t orMask1 = f_group ? 0 : eMask.x;
	const uint64_t orMask2 = f_group ? 0 : eMask.y;
	const uint64_t xexponentMask = (sub & 1) ? eMask.y : eMask.x;

	uint32_t* imm_buf = (uint32_t*)(R + REGISTERS_SIZE / sizeof(uint64_t));
	const uint32_t* compiled_program = (const uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t));

	const uint32_t workers_mask = ((1 << WORKERS_PER_HASH) - 1) << ((threadIdx.x / IDX_WIDTH) * IDX_WIDTH);

	#pragma unroll(1)
	for (int ic = 0; ic < num_iterations; ++ic)
	{
		uint64_t *r, *p0, *p1;
		if ((WORKERS_PER_HASH <= 8) || (sub < 8))
		{
			const uint64_t spMix = *readReg0 ^ *readReg1;
			spAddr0 ^= ((const uint32_t*)&spMix)[0];
			spAddr1 ^= ((const uint32_t*)&spMix)[1];
			spAddr0 &= ScratchpadL3Mask64;
			spAddr1 &= ScratchpadL3Mask64;

			p0 = (uint64_t*)(scratchpad + spAddr0 + sub * 8);
			p1 = (uint64_t*)(scratchpad + spAddr1 + sub * 8);

			r = R + sub;
			*r ^= *p0;

			uint64_t global_mem_data = *p1;
			int32_t* q = (int32_t*)&global_mem_data;

			fe[0] = load_F_E_groups(q[0], andMask, orMask1);
			fe[1] = load_F_E_groups(q[1], andMask, orMask2);
		}

		//if ((global_index == 0) && (ic == 0))
		//{
		//	printf("ic = %d (before)\n", ic);
		//	for (int i = 0; i < 8; ++i)
		//		printf("f%d = %016llx\n", i, bit_cast<uint64_t>(F[i]));
		//	for (int i = 0; i < 8; ++i)
		//		printf("e%d = %016llx\n", i, bit_cast<uint64_t>(E[i]));
		//	printf("\n");
		//}

		if ((WORKERS_PER_HASH == IDX_WIDTH) || (sub < WORKERS_PER_HASH))
			inner_loop<WORKERS_PER_HASH, RANDOMX_FREQ_CFROUND ? -1 : 0, HIGH_PRECISION>(program_length, compiled_program, sub, scratchpad, fp_reg_offset, fp_reg_group_A_offset, R, imm_buf, batch_size, fprc, xexponentMask, workers_mask);

		//if ((global_index == 0) && (ic == RANDOMX_PROGRAM_ITERATIONS - 1))
		//{
		//	printf("ic = %d (after)\n", ic);
		//	for (int i = 0; i < 8; ++i)
		//		printf("r%d = %016llx\n", i, R[i]);
		//	for (int i = 0; i < 8; ++i)
		//		printf("f%d = %016llx\n", i, bit_cast<uint64_t>(F[i]));
		//	for (int i = 0; i < 8; ++i)
		//		printf("e%d = %016llx\n", i, bit_cast<uint64_t>(E[i]));
		//	printf("\n");
		//}

		if ((WORKERS_PER_HASH <= 8) || (sub < 8))
		{
			mx ^= *readReg2 ^ *readReg3;
			mx &= CacheLineAlignMask;

			const uint64_t next_r = *r ^ *(const uint64_t*)(dataset + ma + sub * 8);
			*r = next_r;

			*p1 = next_r;
			*p0 = bit_cast<uint64_t>(f[0]) ^ bit_cast<uint64_t>(e[0]);

			uint32_t tmp = ma;
			ma = mx;
			mx = tmp;

			spAddr0 = 0;
			spAddr1 = 0;
		}
	}

	//if (global_index == 0)
	//{
	//	for (int i = 0; i < 8; ++i)
	//		printf("r%d = %016llx\n", i, R[i]);
	//	for (int i = 0; i < 8; ++i)
	//		printf("fe%d = %016llx\n", i, bit_cast<uint64_t>(F[i]) ^ bit_cast<uint64_t>(E[i]));
	//	printf("\n");
	//}

	if ((WORKERS_PER_HASH > 8) && (sub >= 8))
		return;

	uint64_t* p = ((uint64_t*) vm_states) + idx * (VM_STATE_SIZE / sizeof(uint64_t));
	p[sub] = R[sub];

	if (sub == 0)
	{
		((uint32_t*) rounding)[idx] = fprc;
	}

	if (last)
	{
		p[sub +  8] = bit_cast<uint64_t>(F[sub]) ^ bit_cast<uint64_t>(E[sub]);
		p[sub + 16] = bit_cast<uint64_t>(E[sub]);
	}
	else if (sub == 0)
	{
		((uint32_t*)(p + 16))[0] = ma;
		((uint32_t*)(p + 16))[1] = mx;
	}
}
