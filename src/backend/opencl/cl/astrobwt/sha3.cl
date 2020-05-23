/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#define R64(a,b,c) (((a) << b) | ((a) >> c))

__constant const uint64_t rc[2][ROUNDS] = {
	{0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL,
	0x8000000080008000UL, 0x000000000000808BUL, 0x0000000080000001UL,
	0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008AUL,
	0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
	0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL,
	0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
	0x000000000000800AUL, 0x800000008000000AUL, 0x8000000080008081UL,
	0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL},
	{0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
	0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
	0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL}
};


__constant const int ro[25][2] = {
	{ 0,64}, {44,20}, {43,21}, {21,43}, {14,50},
	{ 1,63}, { 6,58}, {25,39}, { 8,56}, {18,46},
	{62, 2}, {55, 9}, {39,25}, {41,23}, { 2,62},
	{28,36}, {20,44}, { 3,61}, {45,19}, {61, 3},
	{27,37}, {36,28}, {10,54}, {15,49}, {56, 8}
};

__constant const int a[25] = {
	0,  6, 12, 18, 24,
	1,  7, 13, 19, 20,
	2,  8, 14, 15, 21,
	3,  9, 10, 16, 22,
	4,  5, 11, 17, 23
};

__constant const int b[25] = {
	0,  1,  2,  3, 4,
	1,  2,  3,  4, 0,
	2,  3,  4,  0, 1,
	3,  4,  0,  1, 2,
	4,  0,  1,  2, 3
};

__constant const int c[25][3] = {
	{ 0, 1, 2}, { 1, 2, 3}, { 2, 3, 4}, { 3, 4, 0}, { 4, 0, 1},
	{ 5, 6, 7}, { 6, 7, 8}, { 7, 8, 9}, { 8, 9, 5}, { 9, 5, 6},
	{10,11,12}, {11,12,13}, {12,13,14}, {13,14,10}, {14,10,11},
	{15,16,17}, {16,17,18}, {17,18,19}, {18,19,15}, {19,15,16},
	{20,21,22}, {21,22,23}, {22,23,24}, {23,24,20}, {24,20,21}
};

__constant const int d[25] = {
	0,  1,  2,  3,  4,
	10, 11, 12, 13, 14,
	20, 21, 22, 23, 24,
	5,  6,  7,  8,  9,
	15, 16, 17, 18, 19
};

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void sha3(__global const uint8_t* inputs, __global const uint32_t* input_sizes, uint32_t input_stride, __global uint64_t* hashes)
{
	const uint32_t t = get_local_id(0);
	const uint32_t g = get_group_id(0);

	if (t >= 25)
		return;

	const uint32_t s = t % 5;

	const uint64_t input_offset = ((uint64_t)input_stride) * g;
	__global uint64_t* input = (__global uint64_t*)(inputs + input_offset);
	const uint32_t input_size = input_sizes[g] + 1;

	__local uint64_t A[25];
	__local uint64_t C[25];
	__local uint64_t D[25];

	A[t] = 0;

	const uint32_t words = input_size / sizeof(uint64_t);
	const uint32_t tail_size = input_size % sizeof(uint64_t);

	uint32_t wordIndex = 0;
	for (uint32_t i = 0; i < words; ++i, ++input)
	{
		A[wordIndex] ^= *input;
		++wordIndex;
		if (wordIndex == 17)
		{
			#pragma unroll(ROUNDS)
			for (int i = 0; i < ROUNDS; ++i)
			{
				C[t] = A[s] ^ A[s+5] ^ A[s+10] ^ A[s+15] ^ A[s+20];
				D[t] = C[b[20+s]] ^ R64(C[b[5+s]], 1, 63);
				C[t] = R64(A[a[t]] ^ D[b[t]], ro[t][0], ro[t][1]);
				A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]);
				A[t] ^= rc[(t == 0) ? 0 : 1][i]; 
			}
			wordIndex = 0;
		}
	}

	uint64_t tail = 0;
	__global const uint8_t* p = (__global const uint8_t*)input;
	for (uint32_t i = 0; i < tail_size; ++i)
	{
		tail |= (uint64_t)(p[i]) << (i * 8);
	}
	A[wordIndex] ^= tail ^ ((uint64_t)(((uint64_t)(0x02 | (1 << 2))) << (tail_size * 8)));
	A[16] ^= 0x8000000000000000UL;

	#pragma unroll(1)
	for (int i = 0; i < ROUNDS; ++i)
	{
		C[t] = A[s] ^ A[s+5] ^ A[s+10] ^ A[s+15] ^ A[s+20];
		D[t] = C[b[20+s]] ^ R64(C[b[5+s]], 1, 63);
		C[t] = R64(A[a[t]] ^ D[b[t]], ro[t][0], ro[t][1]);
		A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]);
		A[t] ^= rc[(t == 0) ? 0 : 1][i]; 
	}

	if (t < 4)
	{
		hashes += g * (32 / sizeof(uint64_t));
		hashes[t] = A[t];
	}
}

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void sha3_initial(__global const uint8_t* input_data, uint32_t input_size, uint32_t nonce, __global uint64_t* hashes)
{
	const uint32_t t = get_local_id(0);
	const uint32_t g = get_group_id(0);

	if (t >= 25)
		return;

	const uint32_t s = t % 5;

	__global uint64_t* input = (__global uint64_t*)(input_data);

	__local uint64_t A[25];
	__local uint64_t C[25];
	__local uint64_t D[25];

	A[t] = (t < 16) ? input[t] : 0;

	__local uint32_t* nonce_pos = (__local uint32_t*)(A) + 9;
	nonce += g;
	nonce_pos[0] = (nonce_pos[0] & 0xFFFFFFU) | ((nonce & 0xFF) << 24);
	nonce_pos[1] = (nonce_pos[1] & 0xFF000000U) | (nonce >> 8);

	uint32_t wordIndex = input_size / sizeof(uint64_t);
	const uint32_t tail_size = input_size % sizeof(uint64_t);

	A[wordIndex] ^= (uint64_t)(((uint64_t)(0x02 | (1 << 2))) << (tail_size * 8));
	A[16] ^= 0x8000000000000000UL;

	#pragma unroll(ROUNDS)
	for (int i = 0; i < ROUNDS; ++i)
	{
		C[t] = A[s] ^ A[s+5] ^ A[s+10] ^ A[s+15] ^ A[s+20];
		D[t] = C[b[20+s]] ^ R64(C[b[5+s]], 1, 63);
		C[t] = R64(A[a[t]] ^ D[b[t]], ro[t][0], ro[t][1]);
		A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]);
		A[t] ^= rc[(t == 0) ? 0 : 1][i]; 
	}

	if (t < 4)
	{
		hashes += g * (32 / sizeof(uint64_t));
		hashes[t] = A[t];
	}
}
