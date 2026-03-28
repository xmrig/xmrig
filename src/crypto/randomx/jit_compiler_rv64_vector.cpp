/*
Copyright (c) 2018-2020, tevador    <tevador@gmail.com>
Copyright (c) 2019-2021, XMRig      <https://github.com/xmrig>, <support@xmrig.com>
Copyright (c) 2025, SChernykh       <https://github.com/SChernykh>

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

#include "crypto/randomx/configuration.h"
#include "crypto/randomx/jit_compiler_rv64_vector.h"
#include "crypto/randomx/jit_compiler_rv64_vector_static.h"
#include "crypto/randomx/reciprocal.h"
#include "crypto/randomx/superscalar.hpp"
#include "crypto/randomx/program.hpp"
#include "crypto/randomx/soft_aes.h"
#include "backend/cpu/Cpu.h"

namespace randomx {

#define ADDR(x) ((uint8_t*) &(x))
#define DIST(x, y) (ADDR(y) - ADDR(x))

#define JUMP(offset) (0x6F | (((offset) & 0x7FE) << 20) | (((offset) & 0x800) << 9) | ((offset) & 0xFF000))

void* generateDatasetInitVectorRV64(uint8_t* buf, SuperscalarProgram* programs, size_t num_programs)
{
	uint8_t* p = buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_sshash_generated_instructions);

	uint8_t* literals = buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_sshash_imul_rcp_literals);
	uint8_t* cur_literal = literals;

	for (size_t i = 0; i < num_programs; ++i) {
		// Step 4
		size_t k = DIST(randomx_riscv64_vector_sshash_cache_prefetch, randomx_riscv64_vector_sshash_xor);
		memcpy(p, reinterpret_cast<void*>(randomx_riscv64_vector_sshash_cache_prefetch), k);
		p += k;

		// Step 5
		for (uint32_t j = 0; j < programs[i].size; ++j) {
			const uint32_t dst = programs[i].programBuffer[j].dst & 7;
			const uint32_t src = programs[i].programBuffer[j].src & 7;
			const uint32_t modShift = (programs[i].programBuffer[j].mod >> 2) & 3;
			const uint32_t imm32 = programs[i].programBuffer[j].imm32;

			uint32_t inst;
			#define EMIT(data) inst = (data); memcpy(p, &inst, 4); p += 4

			switch (static_cast<SuperscalarInstructionType>(programs[i].programBuffer[j].opcode)) {
			case SuperscalarInstructionType::ISUB_R: 
				// 57 00 00 0A	vsub.vv v0, v0, v0
				EMIT(0x0A000057 | (dst << 7) | (src << 15) | (dst << 20));
				break;

			case SuperscalarInstructionType::IXOR_R:
				// 57 00 00 2E	vxor.vv v0, v0, v0
				EMIT(0x2E000057 | (dst << 7) | (src << 15) | (dst << 20));
				break;

			case SuperscalarInstructionType::IADD_RS:
				if (modShift == 0) {
					// 57 00 00 02	vadd.vv v0, v0, v0
					EMIT(0x02000057 | (dst << 7) | (src << 15) | (dst << 20));
				}
				else {
					// 57 39 00 96	vsll.vi v18, v0, 0
					// 57 00 09 02	vadd.vv v0, v0, v18
					EMIT(0x96003957 | (modShift << 15) | (src << 20));
					EMIT(0x02090057 | (dst << 7) | (dst << 20));
				}
				break;

			case SuperscalarInstructionType::IMUL_R:
				// 57 20 00 96	vmul.vv v0, v0, v0
				EMIT(0x96002057 | (dst << 7) | (src << 15) | (dst << 20));
				break;

			case SuperscalarInstructionType::IROR_C:
				{
#ifdef __riscv_zvkb
					// 57 30 00 52 		vror.vi v0, v0, 0
					EMIT(0x52003057 | (dst << 7) | (dst << 20) | ((imm32 & 31) << 15) | ((imm32 & 32) << 21));
#else // __riscv_zvkb
					const uint32_t shift_right = imm32 & 63;
					const uint32_t shift_left = 64 - shift_right;

					if (shift_right < 32) {
						// 57 39 00 A2	vsrl.vi v18, v0, 0
						EMIT(0xA2003957 | (shift_right << 15) | (dst << 20));
					}
					else {
						// 93 02 00 00	li x5, 0
						// 57 C9 02 A2	vsrl.vx v18, v0, x5
						EMIT(0x00000293 | (shift_right << 20));
						EMIT(0xA202C957 | (dst << 20));
					}

					if (shift_left < 32) {
						// 57 30 00 96	vsll.vi v0, v0, 0
						EMIT(0x96003057 | (dst << 7) | (shift_left << 15) | (dst << 20));
					}
					else {
						// 93 02 00 00	li x5, 0
						// 57 C0 02 96	vsll.vx v0, v0, x5
						EMIT(0x00000293 | (shift_left << 20));
						EMIT(0x9602C057 | (dst << 7) | (dst << 20));
					}

					// 57 00 20 2B vor.vv v0, v18, v0
					EMIT(0x2B200057 | (dst << 7) | (dst << 15));
#endif // __riscv_zvkb
				}
				break;

			case SuperscalarInstructionType::IADD_C7:
			case SuperscalarInstructionType::IADD_C8:
			case SuperscalarInstructionType::IADD_C9:
				// B7 02 00 00	lui x5, 0
				// 9B 82 02 00	addiw x5, x5, 0
				// 57 C0 02 02	vadd.vx v0, v0, x5
				EMIT(0x000002B7 | ((imm32 + ((imm32 & 0x800) << 1)) & 0xFFFFF000));
				EMIT(0x0002829B | ((imm32 & 0x00000FFF) << 20));
				EMIT(0x0202C057 | (dst << 7) | (dst << 20));
				break;

			case SuperscalarInstructionType::IXOR_C7:
			case SuperscalarInstructionType::IXOR_C8:
			case SuperscalarInstructionType::IXOR_C9:
				// B7 02 00 00	lui x5, 0
				// 9B 82 02 00	addiw x5, x5, 0
				// 57 C0 02 2E	vxor.vx v0, v0, x5
				EMIT(0x000002B7 | ((imm32 + ((imm32 & 0x800) << 1)) & 0xFFFFF000));
				EMIT(0x0002829B | ((imm32 & 0x00000FFF) << 20));
				EMIT(0x2E02C057 | (dst << 7) | (dst << 20));
				break;

			case SuperscalarInstructionType::IMULH_R:
				// 57 20 00 92	vmulhu.vv v0, v0, v0
				EMIT(0x92002057 | (dst << 7) | (src << 15) | (dst << 20));
				break;

			case SuperscalarInstructionType::ISMULH_R:
				// 57 20 00 9E	vmulh.vv v0, v0, v0
				EMIT(0x9E002057 | (dst << 7) | (src << 15) | (dst << 20));
				break;

			case SuperscalarInstructionType::IMUL_RCP:
				{
					uint32_t offset = cur_literal - literals;

					if (offset == 2040) {
						literals += 2040;
						offset = 0;

						// 93 87 87 7F	add x15, x15, 2040
						EMIT(0x7F878793);
					}

					const uint64_t r = randomx_reciprocal_fast(imm32);
					memcpy(cur_literal, &r, 8);
					cur_literal += 8;

					// 83 B2 07 00	ld x5, (x15)
					// 57 E0 02 96	vmul.vx v0, v0, x5
					EMIT(0x0007B283 | (offset << 20));
					EMIT(0x9602E057 | (dst << 7) | (dst << 20));
				}
				break;

			default:
				UNREACHABLE;
			}
		}

		// Step 6
		k = DIST(randomx_riscv64_vector_sshash_xor, randomx_riscv64_vector_sshash_end);
		memcpy(p, reinterpret_cast<void*>(randomx_riscv64_vector_sshash_xor), k);
		p += k;

		// Step 7. Set cacheIndex to the value of the register that has the longest dependency chain in the SuperscalarHash function executed in step 5.
		if (i + 1 < num_programs) {
			// vmv.v.v v9, v0 + programs[i].getAddressRegister()
			const uint32_t t = 0x5E0004D7 + (static_cast<uint32_t>(programs[i].getAddressRegister()) << 15);
			memcpy(p, &t, 4);
			p += 4;
		}
	}

	// Emit "J randomx_riscv64_vector_sshash_generated_instructions_end" instruction
	const uint8_t* e = buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_sshash_generated_instructions_end);
	const uint32_t j = JUMP(e - p);
	memcpy(p, &j, 4);

	char* result = (char*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_sshash_dataset_init));

#ifdef __GNUC__
	__builtin___clear_cache(result, (char*)(buf + DIST(randomx_riscv64_vector_sshash_begin, randomx_riscv64_vector_sshash_end)));
#endif

	return result;
}

#define emit16(value) { const uint16_t t = value; memcpy(p, &t, 2); p += 2; }
#define emit32(value) { const uint32_t t = value; memcpy(p, &t, 4); p += 4; }
#define emit64(value) { const uint64_t t = value; memcpy(p, &t, 8); p += 8; }
#define emit_data(arr) { memcpy(p, arr, sizeof(arr)); p += sizeof(arr); }

static void imm_to_x5(uint32_t imm, uint8_t*& p)
{
	const uint32_t imm_hi = (imm + ((imm & 0x800) << 1)) & 0xFFFFF000U;
	const uint32_t imm_lo = imm & 0x00000FFFU;

	if (imm_hi == 0) {
		// li x5, imm_lo
		emit32(0x00000293 + (imm_lo << 20));
		return;
	}

	if (imm_lo == 0) {
		// lui x5, imm_hi
		emit32(0x000002B7 + imm_hi);
		return;
	}

	if (imm_hi < (32 << 12)) {
		//c.lui x5, imm_hi
		emit16(0x6281 + (imm_hi >> 10));
	}
	else {
		// lui x5, imm_hi
		emit32(0x000002B7 + imm_hi);
	}

	// addiw x5, x5, imm_lo
	emit32(0x0002829B | (imm_lo << 20));
}

static void loadFromScratchpad(uint32_t src, uint32_t dst, uint32_t mod, uint32_t imm, uint8_t*& p)
{
	if (src == dst) {
		imm &= RandomX_CurrentConfig.ScratchpadL3Mask_Calculated;

		if (imm <= 2047) {
			// ld x5, imm(x12)
			emit32(0x00063283 | (imm << 20));
		}
		else if (imm <= 2047 * 2) {
			// addi x5, x12, 2047
			emit32(0x7FF60293);
			// ld x5, (imm - 2047)(x5)
			emit32(0x0002B283 | ((imm - 2047) << 20));
		}
		else {
			// lui x5, imm & 0xFFFFF000U
			emit32(0x000002B7 | ((imm + ((imm & 0x800) << 1)) & 0xFFFFF000U));
			// c.add x5, x12
			emit16(0x92B2);
			// ld x5, (imm & 0xFFF)(x5)
			emit32(0x0002B283 | ((imm & 0xFFF) << 20));
		}

		return;
	}

	uint32_t shift = 32;
	uint32_t mask_reg;

	if ((mod & 3) == 0) {
		shift -= RandomX_CurrentConfig.Log2_ScratchpadL2;
		mask_reg = 17;
	}
	else {
		shift -= RandomX_CurrentConfig.Log2_ScratchpadL1;
		mask_reg = 16;
	}

	imm = static_cast<uint32_t>(static_cast<int32_t>(imm << shift) >> shift);

	// 0-0x7FF, 0xFFFFF800-0xFFFFFFFF fit into 12 bit (a single addi instruction)
	if (imm - 0xFFFFF800U < 0x1000U) {
		// addi x5, x20 + src, imm
		emit32(0x000A0293 + (src << 15) + (imm << 20));
	}
	else {
		imm_to_x5(imm, p);
		// c.add x5, x20 + src
		emit16(0x92D2 + (src << 2));
	}

	// and x5, x5, mask_reg
	emit32(0x0002F2B3 + (mask_reg << 20));
	// c.add x5, x12
	emit16(0x92B2);
	// ld x5, 0(x5)
	emit32(0x0002B283);
}

void* generateProgramVectorRV64(uint8_t* buf, Program& prog, ProgramConfiguration& pcfg, const uint8_t (&inst_map)[256], void* entryDataInitScalar, uint32_t datasetOffset)
{
	uint64_t* params = (uint64_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_params));

	params[0] = RandomX_CurrentConfig.ScratchpadL1_Size - 8;
	params[1] = RandomX_CurrentConfig.ScratchpadL2_Size - 8;
	params[2] = RandomX_CurrentConfig.ScratchpadL3_Size - 8;
	params[3] = RandomX_CurrentConfig.DatasetBaseSize - 64;
	params[4] = (1 << RandomX_ConfigurationBase::JumpBits) - 1;

	const bool hasAES = xmrig::Cpu::info()->hasAES();

	if (RandomX_CurrentConfig.Tweak_V2_AES && !hasAES) {
		params[5] = (uint64_t) &lutEnc[2][0];
		params[6] = (uint64_t) &lutDec[2][0];
		params[7] = (uint64_t) lutEncIndex;
		params[8] = (uint64_t) lutDecIndex;

		uint32_t* p1 = (uint32_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_v2_soft_aes_init));

		// Restore vsetivli zero, 4, e32, m1, ta, ma
		*p1 = 0xCD027057;
	}
	else {
		uint32_t* p1 = (uint32_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_v2_soft_aes_init));

		// Emit "J randomx_riscv64_vector_program_main_loop" instruction
		*p1 = JUMP(DIST(randomx_riscv64_vector_program_v2_soft_aes_init, randomx_riscv64_vector_program_main_loop));
	}

	uint64_t* imul_rcp_literals = (uint64_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_imul_rcp_literals));
	uint64_t* cur_literal = imul_rcp_literals;

	uint32_t* spaddr_xor	= (uint32_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_main_loop_spaddr_xor));
	uint32_t* spaddr_xor2	= (uint32_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_scratchpad_prefetch));
	uint32_t* mx_xor	= (uint32_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_main_loop_mx_xor));
	uint32_t* mx_xor_light	= (uint32_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_main_loop_mx_xor_light_mode));

	*spaddr_xor			= 0x014A47B3 + (pcfg.readReg0 << 15) + (pcfg.readReg1 << 20);	// xor x15, readReg0, readReg1
	*spaddr_xor2			= 0x014A42B3 + (pcfg.readReg0 << 15) + (pcfg.readReg1 << 20);	// xor x5,  readReg0, readReg1
	const uint32_t mx_xor_value	= 0x014A42B3 + (pcfg.readReg2 << 15) + (pcfg.readReg3 << 20);	// xor x5,  readReg2, readReg3

	*mx_xor = mx_xor_value;
	*mx_xor_light = mx_xor_value;

	// "slli x5, x5, 32" for RandomX v2, "nop" for RandomX v1
	const uint16_t mp_reg_value = RandomX_CurrentConfig.Tweak_V2_PREFETCH ? 0x1282 : 0x0001;

	memcpy(((uint8_t*)mx_xor) + 8, &mp_reg_value, sizeof(mp_reg_value));
	memcpy(((uint8_t*)mx_xor_light) + 8, &mp_reg_value, sizeof(mp_reg_value));

	// "srli x5, x14, 32" for RandomX v2, "srli x5, x14, 0" for RandomX v1
	const uint32_t mp_reg_value2 = RandomX_CurrentConfig.Tweak_V2_PREFETCH ? 0x02075293 : 0x00075293;
	memcpy(((uint8_t*)mx_xor) + 14, &mp_reg_value2, sizeof(mp_reg_value2));

	if (entryDataInitScalar) {
		void* light_mode_data = buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_main_loop_light_mode_data);

		const uint64_t data[2] = { reinterpret_cast<uint64_t>(entryDataInitScalar), datasetOffset };
		memcpy(light_mode_data, &data, sizeof(data));
	}

	uint8_t* p = (uint8_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_main_loop_instructions));

	// 57C8025E 		vmv.v.x v16, x5
	// 57A9034B 		vsext.vf2 v18, v16
	// 5798214B 		vfcvt.f.x.v v16, v18
	static constexpr uint8_t group_f_convert[] = {
		0x57, 0xC8, 0x02, 0x5E, 0x57, 0xA9, 0x03, 0x4B, 0x57, 0x98, 0x21, 0x4B
	};

	// 57080627 		vand.vv v16, v16, v12
	// 5788062B 		vor.vv v16, v16, v13
	static constexpr uint8_t group_e_post_process[] = { 0x57, 0x08, 0x06, 0x27, 0x57, 0x88, 0x06, 0x2B };

	uint8_t* last_modified[RegistersCount] = { p, p, p, p, p, p, p, p };

	for (uint32_t i = 0, n = prog.getSize(); i < n; ++i) {
		Instruction instr = prog(i);

		uint32_t src = instr.src % RegistersCount;
		uint32_t dst = instr.dst % RegistersCount;
		const uint32_t shift = instr.getModShift();
		uint32_t imm = instr.getImm32();
		const uint32_t mod = instr.mod;

		switch (static_cast<InstructionType>(inst_map[instr.opcode])) {
		case InstructionType::IADD_RS:
			if (shift == 0) {
				// c.add x20 + dst, x20 + src
				emit16(0x9A52 + (src << 2) + (dst << 7));
			}
			else {
#ifdef __riscv_zba
				// sh{shift}add x20 + dst, x20 + src, x20 + dst
				emit32(0x214A0A33 + (shift << 13) + (dst << 7) + (src << 15) + (dst << 20));
#else // __riscv_zba
				// slli x5, x20 + src, shift
				emit32(0x000A1293 + (src << 15) + (shift << 20));
				// c.add x20 + dst, x5
				emit16(0x9A16 + (dst << 7));
#endif // __riscv_zba
			}
			if (dst == RegisterNeedsDisplacement) {
				imm_to_x5(imm, p);

				// c.add x20 + dst, x5
				emit16(0x9A16 + (dst << 7));
			}

			last_modified[dst] = p;
			break;

		case InstructionType::IADD_M:
			loadFromScratchpad(src, dst, mod, imm, p);
			// c.add x20 + dst, x5
			emit16(0x9A16 + (dst << 7));

			last_modified[dst] = p;
			break;

		case InstructionType::ISUB_R:
			if (src != dst) {
				// sub x20 + dst, x20 + dst, x20 + src
				emit32(0x414A0A33 + (dst << 7) + (dst << 15) + (src << 20));
			}
			else {
				imm_to_x5(-imm, p);
				// c.add x20 + dst, x5
				emit16(0x9A16 + (dst << 7));
			}

			last_modified[dst] = p;
			break;

		case InstructionType::ISUB_M:
			loadFromScratchpad(src, dst, mod, imm, p);
			// sub x20 + dst, x20 + dst, x5
			emit32(0x405A0A33 + (dst << 7) + (dst << 15));

			last_modified[dst] = p;
			break;

		case InstructionType::IMUL_R:
			if (src != dst) {
				// mul x20 + dst, x20 + dst, x20 + src
				emit32(0x034A0A33 + (dst << 7) + (dst << 15) + (src << 20));
			}
			else {
				imm_to_x5(imm, p);
				// mul x20 + dst, x20 + dst, x5
				emit32(0x025A0A33 + (dst << 7) + (dst << 15));
			}

			last_modified[dst] = p;
			break;

		case InstructionType::IMUL_M:
			loadFromScratchpad(src, dst, mod, imm, p);
			// mul x20 + dst, x20 + dst, x5
			emit32(0x025A0A33 + (dst << 7) + (dst << 15));

			last_modified[dst] = p;
			break;

		case InstructionType::IMULH_R:
			// mulhu x20 + dst, x20 + dst, x20 + src
			emit32(0x034A3A33 + (dst << 7) + (dst << 15) + (src << 20));

			last_modified[dst] = p;
			break;

		case InstructionType::IMULH_M:
			loadFromScratchpad(src, dst, mod, imm, p);
			// mulhu x20 + dst, x20 + dst, x5
			emit32(0x025A3A33 + (dst << 7) + (dst << 15));

			last_modified[dst] = p;
			break;

		case InstructionType::ISMULH_R:
			// mulh x20 + dst, x20 + dst, x20 + src
			emit32(0x034A1A33 + (dst << 7) + (dst << 15) + (src << 20));

			last_modified[dst] = p;
			break;

		case InstructionType::ISMULH_M:
			loadFromScratchpad(src, dst, mod, imm, p);
			// mulh x20 + dst, x20 + dst, x5
			emit32(0x025A1A33 + (dst << 7) + (dst << 15));

			last_modified[dst] = p;
			break;

		case InstructionType::IMUL_RCP:
			if (!isZeroOrPowerOf2(imm)) {
				const uint64_t offset = (cur_literal - imul_rcp_literals) * 8;
				*(cur_literal++) = randomx_reciprocal_fast(imm);

				static constexpr uint32_t rcp_regs[26] = {
					/* Integer */ 8, 10, 28, 29, 30, 31,
					/* Float   */ 0,  1,  2,  3,  4,  5,  6,  7, 10, 11, 12, 13, 14, 15, 16, 17, 28, 29, 30, 31
				};

				if (offset < 6 * 8) {
					// mul x20 + dst, x20 + dst, rcp_reg
					emit32(0x020A0A33 + (dst << 7) + (dst << 15) + (rcp_regs[offset / 8] << 20));
				}
				else if (offset < 26 * 8) {
					// fmv.x.d x5, rcp_reg
					emit32(0xE20002D3 + (rcp_regs[offset / 8] << 15));
					// mul x20 + dst, x20 + dst, x5
					emit32(0x025A0A33 + (dst << 7) + (dst << 15));
				}
				else {
					// ld x5, offset(x18)
					emit32(0x00093283 + (offset << 20));
					// mul x20 + dst, x20 + dst, x5
					emit32(0x025A0A33 + (dst << 7) + (dst << 15));
				}

				last_modified[dst] = p;
			}
			break;

		case InstructionType::INEG_R:
			// sub x20 + dst, x0, x20 + dst
			emit32(0x41400A33 + (dst << 7) + (dst << 20));

			last_modified[dst] = p;
			break;

		case InstructionType::IXOR_R:
			if (src != dst) {
				// xor x20 + dst, x20 + dst, x20 + src
				emit32(0x014A4A33 + (dst << 7) + (dst << 15) + (src << 20));
			}
			else {
				imm_to_x5(imm, p);
				// xor x20, x20, x5
				emit32(0x005A4A33 + (dst << 7) + (dst << 15));
			}

			last_modified[dst] = p;
			break;

		case InstructionType::IXOR_M:
			loadFromScratchpad(src, dst, mod, imm, p);
			// xor x20, x20, x5
			emit32(0x005A4A33 + (dst << 7) + (dst << 15));

			last_modified[dst] = p;
			break;

#ifdef __riscv_zbb
		case InstructionType::IROR_R:
			if (src != dst) {
				// ror x20 + dst, x20 + dst, x20 + src
				emit32(0x614A5A33 + (dst << 7) + (dst << 15) + (src << 20));
			}
			else {
				// rori x20 + dst, x20 + dst, imm
				emit32(0x600A5A13 + (dst << 7) + (dst << 15) + ((imm & 63) << 20));
			}

			last_modified[dst] = p;
			break;

		case InstructionType::IROL_R:
			if (src != dst) {
				// rol x20 + dst, x20 + dst, x20 + src
				emit32(0x614A1A33 + (dst << 7) + (dst << 15) + (src << 20));
			}
			else {
				// rori x20 + dst, x20 + dst, -imm
				emit32(0x600A5A13 + (dst << 7) + (dst << 15) + ((-imm & 63) << 20));
			}

			last_modified[dst] = p;
			break;
#else // __riscv_zbb
		case InstructionType::IROR_R:
			if (src != dst) {
				// sub x5, x0, x20 + src
				emit32(0x414002B3 + (src << 20));
				// srl x6, x20 + dst, x20 + src
				emit32(0x014A5333 + (dst << 15) + (src << 20));
				// sll x20 + dst, x20 + dst, x5
				emit32(0x005A1A33 + (dst << 7) + (dst << 15));
				// or x20 + dst, x20 + dst, x6
				emit32(0x006A6A33 + (dst << 7) + (dst << 15));
			}
			else {
				// srli x5, x20 + dst, imm
				emit32(0x000A5293 + (dst << 15) + ((imm & 63) << 20));
				// slli x6, x20 + dst, -imm
				emit32(0x000A1313 + (dst << 15) + ((-imm & 63) << 20));
				// or x20 + dst, x5, x6
				emit32(0x0062EA33 + (dst << 7));
			}

			last_modified[dst] = p;
			break;

		case InstructionType::IROL_R:
			if (src != dst) {
				// sub x5, x0, x20 + src
				emit32(0x414002B3 + (src << 20));
				// sll x6, x20 + dst, x20 + src
				emit32(0x014A1333 + (dst << 15) + (src << 20));
				// srl x20 + dst, x20 + dst, x5
				emit32(0x005A5A33 + (dst << 7) + (dst << 15));
				// or x20 + dst, x20 + dst, x6
				emit32(0x006A6A33 + (dst << 7) + (dst << 15));
			}
			else {
				// srli x5, x20 + dst, -imm
				emit32(0x000A5293 + (dst << 15) + ((-imm & 63) << 20));
				// slli x6, x20 + dst, imm
				emit32(0x000A1313 + (dst << 15) + ((imm & 63) << 20));
				// or x20 + dst, x5, x6
				emit32(0x0062EA33 + (dst << 7));
			}

			last_modified[dst] = p;
			break;
#endif // __riscv_zbb

		case InstructionType::ISWAP_R:
			if (src != dst) {
				// c.mv x5, x20 + dst
				emit16(0x82D2 + (dst << 2));
				// c.mv x20 + dst, x20 + src
				emit16(0x8A52 + (src << 2) + (dst << 7));
				// c.mv x20 + src, x5
				emit16(0x8A16 + (src << 7));

				last_modified[src] = p;
				last_modified[dst] = p;
			}
			break;

		case InstructionType::FSWAP_R:
			// vmv.x.s x5, v0 + dst
			emit32(0x420022D7 + (dst << 20));
			// vslide1down.vx v0 + dst, v0 + dst, x5
			emit32(0x3E02E057 + (dst << 7) + (dst << 20));
			break;

		case InstructionType::FADD_R:
			src %= RegisterCountFlt;
			dst %= RegisterCountFlt;

			// vfadd.vv v0 + dst, v0 + dst, v8 + src
			emit32(0x02041057 + (dst << 7) + (src << 15) + (dst << 20));
			break;

		case InstructionType::FADD_M:
			dst %= RegisterCountFlt;

			loadFromScratchpad(src, RegistersCount, mod, imm, p);
			emit_data(group_f_convert);

			// vfadd.vv v0 + dst, v0 + dst, v16
			emit32(0x02081057 + (dst << 7) + (dst << 20));
			break;

		case InstructionType::FSUB_R:
			src %= RegisterCountFlt;
			dst %= RegisterCountFlt;

			// vfsub.vv v0 + dst, v0 + dst, v8 + src
			emit32(0x0A041057 + (dst << 7) + (src << 15) + (dst << 20));
			break;

		case InstructionType::FSUB_M:
			dst %= RegisterCountFlt;

			loadFromScratchpad(src, RegistersCount, mod, imm, p);
			emit_data(group_f_convert);

			// vfsub.vv v0 + dst, v0 + dst, v16
			emit32(0x0A081057 + (dst << 7) + (dst << 20));
			break;

		case InstructionType::FSCAL_R:
			dst %= RegisterCountFlt;

			// vxor.vv v0, v0, v14
			emit32(0x2E070057 + (dst << 7) + (dst << 20));
			break;

		case InstructionType::FMUL_R:
			src %= RegisterCountFlt;
			dst %= RegisterCountFlt;

			// vfmul.vv v4 + dst, v4 + dst, v8 + src
			emit32(0x92441257 + (dst << 7) + (src << 15) + (dst << 20));
			break;

		case InstructionType::FDIV_M:
			dst %= RegisterCountFlt;

			loadFromScratchpad(src, RegistersCount, mod, imm, p);
			emit_data(group_f_convert);
			emit_data(group_e_post_process);

			// vfdiv.vv v0 + dst, v0 + dst, v16
			emit32(0x82481257 + (dst << 7) + (dst << 20));
			break;

		case InstructionType::FSQRT_R:
			dst %= RegisterCountFlt;

			// vfsqrt.v v4 + dst, v4 + dst
			emit32(0x4E401257 + (dst << 7) + (dst << 20));
			break;

		case InstructionType::CBRANCH:
			{
				const uint32_t shift = (mod >> 4) + RandomX_ConfigurationBase::JumpOffset;

				imm |= (1UL << shift);

				if (RandomX_ConfigurationBase::JumpOffset > 0 || shift > 0) {
					imm &= ~(1UL << (shift - 1));
				}

				// slli x6, x7, shift
				// x6 = branchMask
				emit32(0x00039313 + (shift << 20));

				// x5 = imm
				imm_to_x5(imm, p);

				// c.add x20 + dst, x5
				emit16(0x9A16 + (dst << 7));

				// and x5, x20 + dst, x6
				emit32(0x006A72B3 + (dst << 15));

				const int offset = static_cast<int>(last_modified[dst] - p);

				if (offset >= -4096) {
					// beqz x5, offset
					const uint32_t k = static_cast<uint32_t>(offset);
					emit32(0x80028063 | ((k & 0x1E) << 7) | ((k & 0x7E0) << 20) | ((k & 0x800) >> 4));
				}
				else {
					// bnez x5, 8
					emit32(0x00029463);
					// j offset
					const uint32_t k = static_cast<uint32_t>(offset - 4);
					emit32(0x8000006F | ((k & 0x7FE) << 20) | ((k & 0x800) << 9) | (k & 0xFF000));
				}

				for (uint32_t j = 0; j < RegistersCount; ++j) {
					last_modified[j] = p;
				}
			}
			break;

		case InstructionType::CFROUND:
			if ((imm - 1) & 63) {
#ifdef __riscv_zbb
				// rori x5, x20 + src, imm - 1
				emit32(0x600A5293 + (src << 15) + (((imm - 1) & 63) << 20));
#else // __riscv_zbb
				// srli x5, x20 + src, imm - 1
				emit32(0x000A5293 + (src << 15) + (((imm - 1) & 63) << 20));
				// slli x6, x20 + src, 1 - imm
				emit32(0x000A1313 + (src << 15) + (((1 - imm) & 63) << 20));
				// or x5, x5, x6
				emit32(0x0062E2B3);
#endif // __riscv_zbb

				if (RandomX_CurrentConfig.Tweak_V2_CFROUND) {
					// andi x6, x5, 120
					emit32(0x0782F313);
					// bnez x6, +24
					emit32(0x00031C63);
				}

				// andi x5, x5, 6
				emit32(0x0062F293);
			}
			else {
				if (RandomX_CurrentConfig.Tweak_V2_CFROUND) {
					// andi x6, x20 + src, 120
					emit32(0x078A7313 + (src << 15));
					// bnez x6, +24
					emit32(0x00031C63);
				}

				// andi x5, x20 + src, 6
				emit32(0x006A7293 + (src << 15));
			}

			// li x6, 01111000b
			// x6 = CFROUND lookup table
			emit32(0x07800313);
			// srl x5, x6, x5
			emit32(0x005352B3);
			// andi x5, x5, 3
			emit32(0x0032F293);
			// csrw frm, x5
			emit32(0x00229073);
			break;

		case InstructionType::ISTORE:
			{
				uint32_t mask_reg;
				uint32_t shift = 32;

				if ((mod >> 4) >= 14) {
					shift -= RandomX_CurrentConfig.Log2_ScratchpadL3;
					mask_reg = 1; // x1 = L3 mask
				}
				else {
					if ((mod & 3) == 0) {
						shift -= RandomX_CurrentConfig.Log2_ScratchpadL2;
						mask_reg = 17; // x17 = L2 mask
					}
					else {
						shift -= RandomX_CurrentConfig.Log2_ScratchpadL1;
						mask_reg = 16; // x16 = L1 mask
					}
				}

				imm = static_cast<uint32_t>(static_cast<int32_t>(imm << shift) >> shift);
				imm_to_x5(imm, p);

				// c.add x5, x20 + dst
				emit16(0x92D2 + (dst << 2));
				// and x5, x5, x0 + mask_reg
				emit32(0x0002F2B3 + (mask_reg << 20));
				// c.add x5, x12
				emit16(0x92B2);
				// sd x20 + src, 0(x5)
				emit32(0x0142B023 + (src << 20));
			}
			break;

		case InstructionType::NOP:
			break;

		default:
			UNREACHABLE;
		}
	}

	const uint8_t* e;

	if (entryDataInitScalar) {
		// Emit "J randomx_riscv64_vector_program_main_loop_instructions_end_light_mode" instruction
		e = buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_main_loop_instructions_end_light_mode);
	}
	else {
		// Emit "J randomx_riscv64_vector_program_main_loop_instructions_end" instruction
		e = buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_main_loop_instructions_end);
	}

	emit32(JUMP(e - p));

	if (RandomX_CurrentConfig.Tweak_V2_AES) {
		uint32_t* p1 = (uint32_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_main_loop_fe_mix));

		if (hasAES) {
			// Restore vsetivli zero, 4, e32, m1, ta, ma
			*p1 = 0xCD027057;
		}
		else {
			// Emit "J randomx_riscv64_vector_program_main_loop_fe_mix_v2_soft_aes" instruction
			*p1 = JUMP(DIST(randomx_riscv64_vector_program_main_loop_fe_mix, randomx_riscv64_vector_program_main_loop_fe_mix_v2_soft_aes));
		}
	}
	else {
		uint32_t* p1 = (uint32_t*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_main_loop_fe_mix));

		// Emit "J randomx_riscv64_vector_program_main_loop_fe_mix_v1" instruction
		*p1 = JUMP(DIST(randomx_riscv64_vector_program_main_loop_fe_mix, randomx_riscv64_vector_program_main_loop_fe_mix_v1));
	}

#ifdef __GNUC__
	char* p1 = (char*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_params));
	char* p2 = (char*)(buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_end));

	__builtin___clear_cache(p1, p2);
#endif

	return buf + DIST(randomx_riscv64_vector_code_begin, randomx_riscv64_vector_program_begin);
}

} // namespace randomx
