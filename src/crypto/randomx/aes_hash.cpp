/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>

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

#include <thread>
#include <vector>
#include <array>

#include "crypto/randomx/aes_hash.hpp"
#include "base/tools/Chrono.h"
#include "crypto/randomx/randomx.h"
#include "crypto/randomx/soft_aes.h"
#include "crypto/randomx/instruction.hpp"
#include "crypto/randomx/common.hpp"
#include "crypto/rx/Profiler.h"

#define AES_HASH_1R_STATE0 0xd7983aad, 0xcc82db47, 0x9fa856de, 0x92b52c0d
#define AES_HASH_1R_STATE1 0xace78057, 0xf59e125a, 0x15c7b798, 0x338d996e
#define AES_HASH_1R_STATE2 0xe8a07ce4, 0x5079506b, 0xae62c7d0, 0x6a770017
#define AES_HASH_1R_STATE3 0x7e994948, 0x79a10005, 0x07ad828d, 0x630a240c

#define AES_HASH_1R_XKEY0 0x06890201, 0x90dc56bf, 0x8b24949f, 0xf6fa8389
#define AES_HASH_1R_XKEY1 0xed18f99b, 0xee1043c6, 0x51f4e03c, 0x61b263d1

/*
	Calculate a 512-bit hash of 'input' using 4 lanes of AES.
	The input is treated as a set of round keys for the encryption
	of the initial state.

	'inputSize' must be a multiple of 64.

	For a 2 MiB input, this has the same security as 32768-round
	AES encryption.

	Hashing throughput: >20 GiB/s per CPU core with hardware AES
*/
template<int softAes>
void hashAes1Rx4(const void *input, size_t inputSize, void *hash) {
	const uint8_t* inptr = (uint8_t*)input;
	const uint8_t* inputEnd = inptr + inputSize;

	rx_vec_i128 state0, state1, state2, state3;
	rx_vec_i128 in0, in1, in2, in3;

	//intial state
	state0 = rx_set_int_vec_i128(AES_HASH_1R_STATE0);
	state1 = rx_set_int_vec_i128(AES_HASH_1R_STATE1);
	state2 = rx_set_int_vec_i128(AES_HASH_1R_STATE2);
	state3 = rx_set_int_vec_i128(AES_HASH_1R_STATE3);

	//process 64 bytes at a time in 4 lanes
	while (inptr < inputEnd) {
		in0 = rx_load_vec_i128((rx_vec_i128*)inptr + 0);
		in1 = rx_load_vec_i128((rx_vec_i128*)inptr + 1);
		in2 = rx_load_vec_i128((rx_vec_i128*)inptr + 2);
		in3 = rx_load_vec_i128((rx_vec_i128*)inptr + 3);

		state0 = aesenc<softAes>(state0, in0);
		state1 = aesdec<softAes>(state1, in1);
		state2 = aesenc<softAes>(state2, in2);
		state3 = aesdec<softAes>(state3, in3);

		inptr += 64;
	}

	//two extra rounds to achieve full diffusion
	rx_vec_i128 xkey0 = rx_set_int_vec_i128(AES_HASH_1R_XKEY0);
	rx_vec_i128 xkey1 = rx_set_int_vec_i128(AES_HASH_1R_XKEY1);

	state0 = aesenc<softAes>(state0, xkey0);
	state1 = aesdec<softAes>(state1, xkey0);
	state2 = aesenc<softAes>(state2, xkey0);
	state3 = aesdec<softAes>(state3, xkey0);

	state0 = aesenc<softAes>(state0, xkey1);
	state1 = aesdec<softAes>(state1, xkey1);
	state2 = aesenc<softAes>(state2, xkey1);
	state3 = aesdec<softAes>(state3, xkey1);

	//output hash
	rx_store_vec_i128((rx_vec_i128*)hash + 0, state0);
	rx_store_vec_i128((rx_vec_i128*)hash + 1, state1);
	rx_store_vec_i128((rx_vec_i128*)hash + 2, state2);
	rx_store_vec_i128((rx_vec_i128*)hash + 3, state3);
}

template void hashAes1Rx4<false>(const void *input, size_t inputSize, void *hash);
template void hashAes1Rx4<true>(const void *input, size_t inputSize, void *hash);

#define AES_GEN_1R_KEY0 0xb4f44917, 0xdbb5552b, 0x62716609, 0x6daca553
#define AES_GEN_1R_KEY1 0x0da1dc4e, 0x1725d378, 0x846a710d, 0x6d7caf07
#define AES_GEN_1R_KEY2 0x3e20e345, 0xf4c0794f, 0x9f947ec6, 0x3f1262f1
#define AES_GEN_1R_KEY3 0x49169154, 0x16314c88, 0xb1ba317c, 0x6aef8135

/*
	Fill 'buffer' with pseudorandom data based on 512-bit 'state'.
	The state is encrypted using a single AES round per 16 bytes of output
	in 4 lanes.

	'outputSize' must be a multiple of 64.

	The modified state is written back to 'state' to allow multiple
	calls to this function.
*/
template<int softAes>
void fillAes1Rx4(void *state, size_t outputSize, void *buffer) {
	const uint8_t* outptr = (uint8_t*)buffer;
	const uint8_t* outputEnd = outptr + outputSize;

	rx_vec_i128 state0, state1, state2, state3;
	rx_vec_i128 key0, key1, key2, key3;

	key0 = rx_set_int_vec_i128(AES_GEN_1R_KEY0);
	key1 = rx_set_int_vec_i128(AES_GEN_1R_KEY1);
	key2 = rx_set_int_vec_i128(AES_GEN_1R_KEY2);
	key3 = rx_set_int_vec_i128(AES_GEN_1R_KEY3);

	state0 = rx_load_vec_i128((rx_vec_i128*)state + 0);
	state1 = rx_load_vec_i128((rx_vec_i128*)state + 1);
	state2 = rx_load_vec_i128((rx_vec_i128*)state + 2);
	state3 = rx_load_vec_i128((rx_vec_i128*)state + 3);

	while (outptr < outputEnd) {
		state0 = aesdec<softAes>(state0, key0);
		state1 = aesenc<softAes>(state1, key1);
		state2 = aesdec<softAes>(state2, key2);
		state3 = aesenc<softAes>(state3, key3);

		rx_store_vec_i128((rx_vec_i128*)outptr + 0, state0);
		rx_store_vec_i128((rx_vec_i128*)outptr + 1, state1);
		rx_store_vec_i128((rx_vec_i128*)outptr + 2, state2);
		rx_store_vec_i128((rx_vec_i128*)outptr + 3, state3);

		outptr += 64;
	}

	rx_store_vec_i128((rx_vec_i128*)state + 0, state0);
	rx_store_vec_i128((rx_vec_i128*)state + 1, state1);
	rx_store_vec_i128((rx_vec_i128*)state + 2, state2);
	rx_store_vec_i128((rx_vec_i128*)state + 3, state3);
}

template void fillAes1Rx4<true>(void *state, size_t outputSize, void *buffer);
template void fillAes1Rx4<false>(void *state, size_t outputSize, void *buffer);

static constexpr randomx::Instruction inst{ 0xFF, 7, 7, 0xFF, 0xFFFFFFFFU };
alignas(16) static const randomx::Instruction inst_mask[2] = { inst, inst };

template<int softAes>
void fillAes4Rx4(void *state, size_t outputSize, void *buffer) {
	const uint8_t* outptr = (uint8_t*)buffer;
	const uint8_t* outputEnd = outptr + outputSize;

	rx_vec_i128 state0, state1, state2, state3;
	rx_vec_i128 key0, key1, key2, key3, key4, key5, key6, key7;

	key0 = RandomX_CurrentConfig.fillAes4Rx4_Key[0];
	key1 = RandomX_CurrentConfig.fillAes4Rx4_Key[1];
	key2 = RandomX_CurrentConfig.fillAes4Rx4_Key[2];
	key3 = RandomX_CurrentConfig.fillAes4Rx4_Key[3];
	key4 = RandomX_CurrentConfig.fillAes4Rx4_Key[4];
	key5 = RandomX_CurrentConfig.fillAes4Rx4_Key[5];
	key6 = RandomX_CurrentConfig.fillAes4Rx4_Key[6];
	key7 = RandomX_CurrentConfig.fillAes4Rx4_Key[7];

	state0 = rx_load_vec_i128((rx_vec_i128*)state + 0);
	state1 = rx_load_vec_i128((rx_vec_i128*)state + 1);
	state2 = rx_load_vec_i128((rx_vec_i128*)state + 2);
	state3 = rx_load_vec_i128((rx_vec_i128*)state + 3);

#define TRANSFORM do { \
	state0 = aesdec<softAes>(state0, key0); \
	state1 = aesenc<softAes>(state1, key0); \
	state2 = aesdec<softAes>(state2, key4); \
	state3 = aesenc<softAes>(state3, key4); \
	state0 = aesdec<softAes>(state0, key1); \
	state1 = aesenc<softAes>(state1, key1); \
	state2 = aesdec<softAes>(state2, key5); \
	state3 = aesenc<softAes>(state3, key5); \
	state0 = aesdec<softAes>(state0, key2); \
	state1 = aesenc<softAes>(state1, key2); \
	state2 = aesdec<softAes>(state2, key6); \
	state3 = aesenc<softAes>(state3, key6); \
	state0 = aesdec<softAes>(state0, key3); \
	state1 = aesenc<softAes>(state1, key3); \
	state2 = aesdec<softAes>(state2, key7); \
	state3 = aesenc<softAes>(state3, key7); \
} while (0)

	for (int i = 0; i < 2; ++i, outptr += 64) {
		TRANSFORM;
		rx_store_vec_i128((rx_vec_i128*)outptr + 0, state0);
		rx_store_vec_i128((rx_vec_i128*)outptr + 1, state1);
		rx_store_vec_i128((rx_vec_i128*)outptr + 2, state2);
		rx_store_vec_i128((rx_vec_i128*)outptr + 3, state3);
	}

	static_assert(sizeof(inst_mask) == sizeof(rx_vec_i128), "Incorrect inst_mask size");
	const rx_vec_i128 mask = *reinterpret_cast<const rx_vec_i128*>(inst_mask);

	while (outptr < outputEnd) {
		TRANSFORM;
		rx_store_vec_i128((rx_vec_i128*)outptr + 0, rx_and_vec_i128(state0, mask));
		rx_store_vec_i128((rx_vec_i128*)outptr + 1, rx_and_vec_i128(state1, mask));
		rx_store_vec_i128((rx_vec_i128*)outptr + 2, rx_and_vec_i128(state2, mask));
		rx_store_vec_i128((rx_vec_i128*)outptr + 3, rx_and_vec_i128(state3, mask));
		outptr += 64;
	}
}

template void fillAes4Rx4<true>(void *state, size_t outputSize, void *buffer);
template void fillAes4Rx4<false>(void *state, size_t outputSize, void *buffer);

template<int softAes, int unroll>
void hashAndFillAes1Rx4(void *scratchpad, size_t scratchpadSize, void *hash, void* fill_state) {
	PROFILE_SCOPE(RandomX_AES);

	uint8_t* scratchpadPtr = (uint8_t*)scratchpad;
	const uint8_t* scratchpadEnd = scratchpadPtr + scratchpadSize;

	// initial state
	rx_vec_i128 hash_state0 = rx_set_int_vec_i128(AES_HASH_1R_STATE0);
	rx_vec_i128 hash_state1 = rx_set_int_vec_i128(AES_HASH_1R_STATE1);
	rx_vec_i128 hash_state2 = rx_set_int_vec_i128(AES_HASH_1R_STATE2);
	rx_vec_i128 hash_state3 = rx_set_int_vec_i128(AES_HASH_1R_STATE3);

	const rx_vec_i128 key0 = rx_set_int_vec_i128(AES_GEN_1R_KEY0);
	const rx_vec_i128 key1 = rx_set_int_vec_i128(AES_GEN_1R_KEY1);
	const rx_vec_i128 key2 = rx_set_int_vec_i128(AES_GEN_1R_KEY2);
	const rx_vec_i128 key3 = rx_set_int_vec_i128(AES_GEN_1R_KEY3);

	rx_vec_i128 fill_state0 = rx_load_vec_i128((rx_vec_i128*)fill_state + 0);
	rx_vec_i128 fill_state1 = rx_load_vec_i128((rx_vec_i128*)fill_state + 1);
	rx_vec_i128 fill_state2 = rx_load_vec_i128((rx_vec_i128*)fill_state + 2);
	rx_vec_i128 fill_state3 = rx_load_vec_i128((rx_vec_i128*)fill_state + 3);

	constexpr int PREFETCH_DISTANCE = 7168;
	const char* prefetchPtr = ((const char*)scratchpad) + PREFETCH_DISTANCE;
	scratchpadEnd -= PREFETCH_DISTANCE;

	for (int i = 0; i < 2; ++i) {
		//process 64 bytes at a time in 4 lanes
		while (scratchpadPtr < scratchpadEnd) {
#define HASH_STATE(k) \
			hash_state0 = aesenc<softAes>(hash_state0, rx_load_vec_i128((rx_vec_i128*)scratchpadPtr + k * 4 + 0)); \
			hash_state1 = aesdec<softAes>(hash_state1, rx_load_vec_i128((rx_vec_i128*)scratchpadPtr + k * 4 + 1)); \
			hash_state2 = aesenc<softAes>(hash_state2, rx_load_vec_i128((rx_vec_i128*)scratchpadPtr + k * 4 + 2)); \
			hash_state3 = aesdec<softAes>(hash_state3, rx_load_vec_i128((rx_vec_i128*)scratchpadPtr + k * 4 + 3));

#define FILL_STATE(k) \
			fill_state0 = aesdec<softAes>(fill_state0, key0); \
			fill_state1 = aesenc<softAes>(fill_state1, key1); \
			fill_state2 = aesdec<softAes>(fill_state2, key2); \
			fill_state3 = aesenc<softAes>(fill_state3, key3); \
			rx_store_vec_i128((rx_vec_i128*)scratchpadPtr + k * 4 + 0, fill_state0); \
			rx_store_vec_i128((rx_vec_i128*)scratchpadPtr + k * 4 + 1, fill_state1); \
			rx_store_vec_i128((rx_vec_i128*)scratchpadPtr + k * 4 + 2, fill_state2); \
			rx_store_vec_i128((rx_vec_i128*)scratchpadPtr + k * 4 + 3, fill_state3);

			switch (softAes) {
				case 0:
					HASH_STATE(0);
					HASH_STATE(1);

					FILL_STATE(0);
					FILL_STATE(1);

					rx_prefetch_t0(prefetchPtr);
					rx_prefetch_t0(prefetchPtr + 64);

					scratchpadPtr += 128;
					prefetchPtr += 128;

					break;

				default:
					switch (unroll) {
						case 4:
							HASH_STATE(0);
							FILL_STATE(0);
							rx_prefetch_t0(prefetchPtr);

							HASH_STATE(1);
							FILL_STATE(1);
							rx_prefetch_t0(prefetchPtr + 64);

							HASH_STATE(2);
							FILL_STATE(2);
							rx_prefetch_t0(prefetchPtr + 64 * 2);

							HASH_STATE(3);
							FILL_STATE(3);
							rx_prefetch_t0(prefetchPtr + 64 * 3);

							scratchpadPtr += 64 * 4;
							prefetchPtr += 64 * 4;
							break;

						case 2:
							HASH_STATE(0);
							FILL_STATE(0);
							rx_prefetch_t0(prefetchPtr);

							HASH_STATE(1);
							FILL_STATE(1);
							rx_prefetch_t0(prefetchPtr + 64);

							scratchpadPtr += 64 * 2;
							prefetchPtr += 64 * 2;
							break;

						default:
							HASH_STATE(0);
							FILL_STATE(0);
							rx_prefetch_t0(prefetchPtr);

							scratchpadPtr += 64;
							prefetchPtr += 64;

							break;
					}
					break;
			}
		}
		prefetchPtr = (const char*) scratchpad;
		scratchpadEnd += PREFETCH_DISTANCE;
	}

	rx_store_vec_i128((rx_vec_i128*)fill_state + 0, fill_state0);
	rx_store_vec_i128((rx_vec_i128*)fill_state + 1, fill_state1);
	rx_store_vec_i128((rx_vec_i128*)fill_state + 2, fill_state2);
	rx_store_vec_i128((rx_vec_i128*)fill_state + 3, fill_state3);

	//two extra rounds to achieve full diffusion
	rx_vec_i128 xkey0 = rx_set_int_vec_i128(AES_HASH_1R_XKEY0);
	rx_vec_i128 xkey1 = rx_set_int_vec_i128(AES_HASH_1R_XKEY1);

	hash_state0 = aesenc<softAes>(hash_state0, xkey0);
	hash_state1 = aesdec<softAes>(hash_state1, xkey0);
	hash_state2 = aesenc<softAes>(hash_state2, xkey0);
	hash_state3 = aesdec<softAes>(hash_state3, xkey0);

	hash_state0 = aesenc<softAes>(hash_state0, xkey1);
	hash_state1 = aesdec<softAes>(hash_state1, xkey1);
	hash_state2 = aesenc<softAes>(hash_state2, xkey1);
	hash_state3 = aesdec<softAes>(hash_state3, xkey1);

	//output hash
	rx_store_vec_i128((rx_vec_i128*)hash + 0, hash_state0);
	rx_store_vec_i128((rx_vec_i128*)hash + 1, hash_state1);
	rx_store_vec_i128((rx_vec_i128*)hash + 2, hash_state2);
	rx_store_vec_i128((rx_vec_i128*)hash + 3, hash_state3);
}

template void hashAndFillAes1Rx4<0,2>(void* scratchpad, size_t scratchpadSize, void* hash, void* fill_state);
template void hashAndFillAes1Rx4<1,1>(void* scratchpad, size_t scratchpadSize, void* hash, void* fill_state);
template void hashAndFillAes1Rx4<2,1>(void* scratchpad, size_t scratchpadSize, void* hash, void* fill_state);
template void hashAndFillAes1Rx4<2,2>(void* scratchpad, size_t scratchpadSize, void* hash, void* fill_state);
template void hashAndFillAes1Rx4<2,4>(void* scratchpad, size_t scratchpadSize, void* hash, void* fill_state);

hashAndFillAes1Rx4_impl* softAESImpl = &hashAndFillAes1Rx4<1,1>;

void SelectSoftAESImpl(size_t threadsCount)
{
  constexpr uint64_t test_length_ms = 100;
  const std::array<hashAndFillAes1Rx4_impl *, 4> impl = {
    &hashAndFillAes1Rx4<1,1>,
    &hashAndFillAes1Rx4<2,1>,
    &hashAndFillAes1Rx4<2,2>,
    &hashAndFillAes1Rx4<2,4>,
  };
  size_t fast_idx = 0;
  double fast_speed = 0.0;
  for (size_t run = 0; run < 3; ++run) {
    for (size_t i = 0; i < impl.size(); ++i) {
      const double t1 = xmrig::Chrono::highResolutionMSecs();
      std::vector<uint32_t> count(threadsCount, 0);
      std::vector<std::thread> threads;
      for (size_t t = 0; t < threadsCount; ++t) {
        threads.emplace_back([&, t]() {
          std::vector<uint8_t> scratchpad(10 * 1024);
          alignas(16) uint8_t hash[64] = {};
          alignas(16) uint8_t state[64] = {};
          do {
          (*impl[i])(scratchpad.data(), scratchpad.size(), hash, state);
          ++count[t];
          } while (xmrig::Chrono::highResolutionMSecs() - t1 < test_length_ms);
        });
      }
      uint32_t total = 0;
      for (size_t t = 0; t < threadsCount; ++t) {
        threads[t].join();
        total += count[t];
      }
      const double t2 = xmrig::Chrono::highResolutionMSecs();
      const double speed = total * 1e3 / (t2 - t1);
      if (speed > fast_speed) {
        fast_idx = i;
        fast_speed = speed;
      }
    }
  }
  softAESImpl = impl[fast_idx];
}
