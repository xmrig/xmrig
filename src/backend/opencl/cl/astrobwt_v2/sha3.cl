/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#define ROUNDS 24

__constant const uint64_t rc[24] = {
	0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL,
	0x8000000080008000UL, 0x000000000000808BUL, 0x0000000080000001UL,
	0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008AUL,
	0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
	0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL,
	0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
	0x000000000000800AUL, 0x800000008000000AUL, 0x8000000080008081UL,
	0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

__constant const int c[25][2] = {
	{ 1, 2}, { 2, 3}, { 3, 4}, { 4, 0}, { 0, 1},
	{ 6, 7}, { 7, 8}, { 8, 9}, { 9, 5}, { 5, 6},
	{11,12}, {12,13}, {13,14}, {14,10}, {10,11},
	{16,17}, {17,18}, {18,19}, {19,15}, {15,16},
	{21,22}, {22,23}, {23,24}, {24,20}, {20,21}
};

__constant const int ppi[25][2] = {
	{0, 0},  {6, 44},  {12, 43}, {18, 21}, {24, 14}, {3, 28},  {9, 20}, {10, 3}, {16, 45},
	{22, 61}, {1, 1},   {7, 6},   {13, 25}, {19, 8},  {20, 18}, {4, 27}, {5, 36}, {11, 10},
	{17, 15}, {23, 56}, {2, 62},  {8, 55},  {14, 39}, {15, 41}, {21, 2}
};

#define R64(a,b,c) (((a) << b) | ((a) >> c))

#define ROUND(k) \
do { \
	C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20]; \
	A[t] ^= C[s + 4] ^ R64(C[s + 1], 1, 63); \
	C[t] = R64(A[at], ro0, ro1); \
	A[t] = (C[t] ^ ((~C[c1]) & C[c2])) ^ (k1 & (k)); \
} while (0)

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void sha3(__global const uint8_t* inputs, __global uint64_t* hashes)
{
	const uint32_t t = get_local_id(0);

	if (t >= 25) {
		return;
	}

	const uint32_t g = get_group_id(0);
	const uint64_t input_offset = 10240 * 2 * g;
	__global const uint64_t* input = (__global const uint64_t*)(inputs + input_offset);

	__local uint64_t A[25];
	__local uint64_t C[25];

	A[t] = 0;

	const uint32_t s = t % 5;
	const int at = ppi[t][0];
	const int ro0 = ppi[t][1];
	const int ro1 = 64 - ro0;
	const int c1 = c[t][0];
	const int c2 = c[t][1];
	const uint64_t k1 = (t == 0) ? 0xFFFFFFFFFFFFFFFFUL : 0UL;

	const uint32_t input_size = 9973 * 2;
	const uint32_t input_words = input_size / sizeof(uint64_t);
	__global const uint64_t* const input_end17 = input + ((input_words / 17) * 17);
	__global const uint64_t* const input_end = input + input_words;

	for (; input < input_end17; input += 17) {
		if (t < 17) A[t] ^= input[t];

		ROUND(0x0000000000000001UL); ROUND(0x0000000000008082UL); ROUND(0x800000000000808AUL);
		ROUND(0x8000000080008000UL); ROUND(0x000000000000808BUL); ROUND(0x0000000080000001UL);
		ROUND(0x8000000080008081UL); ROUND(0x8000000000008009UL); ROUND(0x000000000000008AUL);
		ROUND(0x0000000000000088UL); ROUND(0x0000000080008009UL); ROUND(0x000000008000000AUL);
		ROUND(0x000000008000808BUL); ROUND(0x800000000000008BUL); ROUND(0x8000000000008089UL);
		ROUND(0x8000000000008003UL); ROUND(0x8000000000008002UL); ROUND(0x8000000000000080UL);
		ROUND(0x000000000000800AUL); ROUND(0x800000008000000AUL); ROUND(0x8000000080008081UL);
		ROUND(0x8000000000008080UL); ROUND(0x0000000080000001UL); ROUND(0x8000000080008008UL);
	}

	const uint32_t wordIndex = input_end - input;
	if (t < wordIndex) A[t] ^= input[t];

	if (t == 0) {
		uint64_t tail = 0;
		__global const uint8_t* p = (__global const uint8_t*)input_end;
		const uint32_t tail_size = input_size % sizeof(uint64_t);
		for (uint32_t i = 0; i < tail_size; ++i) {
			tail |= (uint64_t)(p[i]) << (i * 8);
		}

		A[wordIndex] ^= tail ^ ((uint64_t)(((uint64_t)(0x02 | (1 << 2))) << (tail_size * 8)));
		A[16] ^= 0x8000000000000000UL;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	#pragma unroll 1
	for (int i = 0; i < ROUNDS; ++i) {
		C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
		A[t] ^= C[s + 4] ^ R64(C[s + 1], 1, 63);
		C[t] = R64(A[at], ro0, ro1);
		A[t] = (C[t] ^ ((~C[c1]) & C[c2])) ^ (rc[i] & k1);
	}

	if (t < 4) {
		hashes[g * 4 + t] = A[t];
	}
}

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void sha3_initial(__global const uint8_t* input_data, uint32_t input_size, uint32_t nonce, __global uint64_t* hashes)
{
	const uint32_t t = get_local_id(0);
	const uint32_t g = get_group_id(0);

	if (t >= 25) {
		return;
	}

	__global const uint64_t* input = (__global const uint64_t*)(input_data);

	__local uint64_t A[25];
	__local uint64_t C[25];

	const uint32_t input_words = input_size / sizeof(uint64_t);
	A[t] = (t < input_words) ? input[t] : 0;

	if (t == 0) {
		((__local uint32_t*)A)[11] = nonce + g;

		const uint32_t tail_size = input_size % sizeof(uint64_t);
		A[input_words] ^= (uint64_t)(((uint64_t)(0x02 | (1 << 2))) << (tail_size * 8));
		A[16] ^= 0x8000000000000000UL;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	const uint32_t s = t % 5;
	const int at = ppi[t][0];
	const int ro0 = ppi[t][1];
	const int ro1 = 64 - ro0;
	const int c1 = c[t][0];
	const int c2 = c[t][1];
	const uint64_t k1 = (t == 0) ? (uint64_t)(-1) : 0;

	ROUND(0x0000000000000001UL); ROUND(0x0000000000008082UL); ROUND(0x800000000000808AUL);
	ROUND(0x8000000080008000UL); ROUND(0x000000000000808BUL); ROUND(0x0000000080000001UL);
	ROUND(0x8000000080008081UL); ROUND(0x8000000000008009UL); ROUND(0x000000000000008AUL);
	ROUND(0x0000000000000088UL); ROUND(0x0000000080008009UL); ROUND(0x000000008000000AUL);
	ROUND(0x000000008000808BUL); ROUND(0x800000000000008BUL); ROUND(0x8000000000008089UL);
	ROUND(0x8000000000008003UL); ROUND(0x8000000000008002UL); ROUND(0x8000000000000080UL);
	ROUND(0x000000000000800AUL); ROUND(0x800000008000000AUL); ROUND(0x8000000080008081UL);
	ROUND(0x8000000000008080UL); ROUND(0x0000000080000001UL); ROUND(0x8000000080008008UL);

	if (t < 4) {
		hashes[g * 4 + t] = A[t];
	}
}

#undef ROUNDS
#undef R64
#undef ROUND
