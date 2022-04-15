/* XMRig
 * Copyright (c) 2018      Lee Clagett              <https://github.com/vtnerd>
 * Copyright (c) 2018-2019 tevador                  <tevador@gmail.com>
 * Copyright (c) 2000      Transmeta Corporation    <https://github.com/intel/msr-tools>
 * Copyright (c) 2004-2008 H. Peter Anvin           <https://github.com/intel/msr-tools>
 * Copyright (c) 2018-2021 SChernykh                <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig                    <https://github.com/xmrig>, <support@xmrig.com>
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

#include "crypto/astrobwt/AstroBWT.h"
#include "backend/cpu/Cpu.h"
#include "base/crypto/sha3.h"
#include "base/tools/bswap_64.h"
#include "crypto/cn/CryptoNight.h"
#include "crypto/astrobwt/sort_indices2.h"


#include <limits>


static bool astrobwtInitialized = false;

#ifdef ASTROBWT_AVX2
static bool hasAVX2 = false;

extern "C"
#ifndef _MSC_VER
__attribute__((ms_abi))
#endif
void SHA3_256_AVX2_ASM(const void* in, size_t inBytes, void* out);
#endif

#ifdef XMRIG_ARM
extern "C" {
#include "salsa20_ref/ecrypt-sync.h"
}

static void Salsa20_XORKeyStream(const void* key, void* output, size_t size)
{
	uint8_t iv[8] = {};
	ECRYPT_ctx ctx;
	ECRYPT_keysetup(&ctx, static_cast<const uint8_t*>(key), 256, 64);
	ECRYPT_ivsetup(&ctx, iv);
	ECRYPT_keystream_bytes(&ctx, static_cast<uint8_t*>(output), size);
	memset(static_cast<uint8_t*>(output) - 16, 0, 16);
	memset(static_cast<uint8_t*>(output) + size, 0, 16);
}
#else
#include "Salsa20.hpp"

static void Salsa20_XORKeyStream(const void* key, void* output, size_t size)
{
	const uint64_t iv = 0;
	ZeroTier::Salsa20 s(key, &iv);
	s.XORKeyStream(output, static_cast<uint32_t>(size));
	memset(static_cast<uint8_t*>(output) - 16, 0, 16);
	memset(static_cast<uint8_t*>(output) + size, 0, 16);
}

extern "C" int salsa20_stream_avx2(void* c, uint64_t clen, const void* iv, const void* key);

static void Salsa20_XORKeyStream_AVX256(const void* key, void* output, size_t size)
{
	const uint64_t iv = 0;
	salsa20_stream_avx2(output, size, &iv, key);
	memset(static_cast<uint8_t*>(output) - 16, 0, 16);
	memset(static_cast<uint8_t*>(output) + size, 0, 16);
}
#endif

bool xmrig::astrobwt::astrobwt_dero_v2(const void* input_data, uint32_t input_size, void* scratchpad, uint8_t* output_hash)
{
	constexpr size_t N = 9973;
	constexpr size_t STRIDE = 10240;

	alignas(8) uint8_t key[32];
	uint8_t* scratchpad_ptr = (uint8_t*)(scratchpad) + 64;
	uint8_t* v = scratchpad_ptr;
	uint32_t* indices = (uint32_t*)(scratchpad_ptr + STRIDE);
	uint32_t* tmp_indices = (uint32_t*)(scratchpad_ptr + STRIDE * 5);

#ifdef ASTROBWT_AVX2
	if (hasAVX2) {
		SHA3_256_AVX2_ASM(input_data, input_size, key);
		Salsa20_XORKeyStream_AVX256(key, v, N);
	}
	else
#endif
	{
		sha3_HashBuffer(256, SHA3_FLAGS_NONE, input_data, input_size, key, sizeof(key));
		Salsa20_XORKeyStream(key, v, N);
	}

	sort_indices_astrobwt_v2(N, v, indices, tmp_indices);

#ifdef ASTROBWT_AVX2
	if (hasAVX2) {
		SHA3_256_AVX2_ASM(indices, N * 2, output_hash);
	}
	else
#endif
	{
		sha3_HashBuffer(256, SHA3_FLAGS_NONE, indices, N * 2, output_hash, 32);
	}

	return true;
}


void xmrig::astrobwt::init()
{
	if (!astrobwtInitialized) {
#		ifdef ASTROBWT_AVX2
		hasAVX2 = Cpu::info()->hasAVX2();
#		endif

		astrobwtInitialized = true;
	}
}


template<>
void xmrig::astrobwt::single_hash<xmrig::Algorithm::ASTROBWT_DERO_2>(const uint8_t* input, size_t size, uint8_t* output, cryptonight_ctx** ctx, uint64_t)
{
	astrobwt_dero_v2(input, static_cast<uint32_t>(size), ctx[0]->memory, output);
}
