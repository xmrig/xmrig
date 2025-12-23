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

namespace randomx {

#define ADDR(x) ((uint8_t*) &(x))
#define DIST(x, y) (ADDR(y) - ADDR(x))

void* generateDatasetInitVectorRV64(uint8_t* buf, size_t buf_size, SuperscalarProgram* programs, size_t num_programs)
{
	memcpy(buf, reinterpret_cast<void*>(randomx_riscv64_vector_sshash_begin), buf_size);

	uint8_t* p = buf + DIST(randomx_riscv64_vector_sshash_begin, randomx_riscv64_vector_sshash_generated_instructions);

	uint8_t* literals = buf + DIST(randomx_riscv64_vector_sshash_begin, randomx_riscv64_vector_sshash_imul_rcp_literals);
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
				// 57 39 00 96	vsll.vi v18, v0, 0
				// 57 00 09 02	vadd.vv v0, v0, v18
				EMIT(0x96003957 | (modShift << 15) | (src << 20));
				EMIT(0x02090057 | (dst << 7) | (dst << 20));
				break;

			case SuperscalarInstructionType::IMUL_R:
				// 57 20 00 96	vmul.vv v0, v0, v0
				EMIT(0x96002057 | (dst << 7) | (src << 15) | (dst << 20));
				break;

			case SuperscalarInstructionType::IROR_C:
				{
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
				}
				break;

			case SuperscalarInstructionType::IADD_C7:
			case SuperscalarInstructionType::IADD_C8:
			case SuperscalarInstructionType::IADD_C9:
				// B7 02 00 00	lui x5, 0
				// 9B 82 02 00	addiw x5, x5, 0
				// 57 C0 02 02	vadd.vx v0, v0, x5
				EMIT(0x000002B7 | ((imm32 + ((imm32 & 0x800) << 1)) & 0xFFFFF000));
				EMIT(0x0002829B | ((imm32 & 0x00000FFF)) << 20);
				EMIT(0x0202C057 | (dst << 7) | (dst << 20));
				break;

			case SuperscalarInstructionType::IXOR_C7:
			case SuperscalarInstructionType::IXOR_C8:
			case SuperscalarInstructionType::IXOR_C9:
				// B7 02 00 00	lui x5, 0
				// 9B 82 02 00	addiw x5, x5, 0
				// 57 C0 02 2E	vxor.vx v0, v0, x5
				EMIT(0x000002B7 | ((imm32 + ((imm32 & 0x800) << 1)) & 0xFFFFF000));
				EMIT(0x0002829B | ((imm32 & 0x00000FFF)) << 20);
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
				break;
			}
		}

		// Step 6
		k = DIST(randomx_riscv64_vector_sshash_xor, randomx_riscv64_vector_sshash_set_cache_index);
		memcpy(p, reinterpret_cast<void*>(randomx_riscv64_vector_sshash_xor), k);
		p += k;

		// Step 7
		if (i + 1 < num_programs) {
			memcpy(p, reinterpret_cast<uint8_t*>(randomx_riscv64_vector_sshash_set_cache_index) + programs[i].getAddressRegister() * 4, 4);
			p += 4;
		}
	}

	// Emit "J randomx_riscv64_vector_sshash_generated_instructions_end" instruction
	const uint8_t* e = buf + DIST(randomx_riscv64_vector_sshash_begin, randomx_riscv64_vector_sshash_generated_instructions_end);
	const uint32_t k = e - p;
	const uint32_t j = 0x6F | ((k & 0x7FE) << 20) | ((k & 0x800) << 9) | (k & 0xFF000);
	memcpy(p, &j, 4);

#ifdef __GNUC__
	__builtin___clear_cache((char*) buf, (char*)(buf + buf_size));
#endif

	return buf + DIST(randomx_riscv64_vector_sshash_begin, randomx_riscv64_vector_sshash_dataset_init);
}

} // namespace randomx
