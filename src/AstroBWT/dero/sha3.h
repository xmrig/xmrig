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


#pragma once

namespace AstroBWT_Dero {

constexpr int ROUNDS = 24;

__device__ uint64_t R64(uint64_t a, int b, int c) { return (a << b) | (a >> c); }

__constant__ static const uint64_t rc[2][ROUNDS] = {
	{0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
	0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
	0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
	0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
	0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
	0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
	0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
	0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL},
	{0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
	0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
	0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL}
};


/* Rho-Offsets. Note that for each entry pair their respective sum is 64.
Only the first entry of each pair is a rho-offset. The second part is
used in the R64 macros. */
__constant__ static const int ro[25][2] = {
	/*y=0*/         /*y=1*/         /*y=2*/         /*y=3*/         /*y=4*/
	/*x=0*/{ 0,64}, /*x=1*/{44,20}, /*x=2*/{43,21}, /*x=3*/{21,43}, /*x=4*/{14,50},
	/*x=1*/{ 1,63}, /*x=2*/{ 6,58}, /*x=3*/{25,39}, /*x=4*/{ 8,56}, /*x=0*/{18,46},
	/*x=2*/{62, 2}, /*x=3*/{55, 9}, /*x=4*/{39,25}, /*x=0*/{41,23}, /*x=1*/{ 2,62},
	/*x=3*/{28,36}, /*x=4*/{20,44}, /*x=0*/{ 3,61}, /*x=1*/{45,19}, /*x=2*/{61, 3},
	/*x=4*/{27,37}, /*x=0*/{36,28}, /*x=1*/{10,54}, /*x=2*/{15,49}, /*x=3*/{56, 8}
};

__constant__ static const int a[25] = {
	0,  6, 12, 18, 24,
	1,  7, 13, 19, 20,
	2,  8, 14, 15, 21,
	3,  9, 10, 16, 22,
	4,  5, 11, 17, 23
};

__constant__ static const int b[25] = {
	0,  1,  2,  3, 4,
	1,  2,  3,  4, 0,
	2,  3,  4,  0, 1,
	3,  4,  0,  1, 2,
	4,  0,  1,  2, 3
};

__constant__ static const int c[25][3] = {
	{ 0, 1, 2}, { 1, 2, 3}, { 2, 3, 4}, { 3, 4, 0}, { 4, 0, 1},
	{ 5, 6, 7}, { 6, 7, 8}, { 7, 8, 9}, { 8, 9, 5}, { 9, 5, 6},
	{10,11,12}, {11,12,13}, {12,13,14}, {13,14,10}, {14,10,11},
	{15,16,17}, {16,17,18}, {17,18,19}, {18,19,15}, {19,15,16},
	{20,21,22}, {21,22,23}, {22,23,24}, {23,24,20}, {24,20,21}
};

__constant__ static const int d[25] = {
	0,  1,  2,  3,  4,
	10, 11, 12, 13, 14,
	20, 21, 22, 23, 24,
	5,  6,  7,  8,  9,
	15, 16, 17, 18, 19
};

__global__ void __launch_bounds__(32) sha3(const uint8_t* inputs, const uint32_t* input_sizes, uint32_t input_stride, uint64_t* hashes)
{
	const uint32_t t = threadIdx.x;
	const uint32_t g = blockIdx.x;

	if (t >= 25)
		return;

	const uint32_t s = t % 5;

	const uint64_t input_offset = ((uint64_t)input_stride) * g;
	const uint64_t* input = (const uint64_t*)(inputs + input_offset);
	const uint32_t input_size = input_sizes[g] + 1;

	__shared__ uint64_t A[25];
	__shared__ uint64_t C[25];
	__shared__ uint64_t D[25];

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
				C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
				D[t] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
				C[t] = R64(A[a[t]] ^ D[b[t]], ro[t][0], ro[t][1]);
				A[d[t]] = C[c[t][0]] ^ ((~C[c[t][1]]) & C[c[t][2]]);
				A[t] ^= rc[(t == 0) ? 0 : 1][i];
			}
			wordIndex = 0;
		}
	}

	uint64_t tail = 0;
	const uint8_t* p = (const uint8_t*)input;
	for (uint32_t i = 0; i < tail_size; ++i)
	{
		tail |= (uint64_t)(p[i]) << (i * 8);
	}
	A[wordIndex] ^= tail ^ ((uint64_t)(((uint64_t)(0x02 | (1 << 2))) << (tail_size * 8)));
	A[16] ^= 0x8000000000000000ULL;

	#pragma unroll(1)
	for (int i = 0; i < ROUNDS; ++i)
	{
		C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
		D[t] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
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

__global__ void __launch_bounds__(32) sha3_initial(const uint8_t* input_data, uint32_t input_size, uint32_t nonce, uint64_t* hashes)
{
	const uint32_t t = threadIdx.x;
	const uint32_t g = blockIdx.x;

	if (t >= 25)
		return;

	const uint32_t s = t % 5;

	const uint64_t* input = (const uint64_t*)(input_data);

	__shared__ uint64_t A[25];
	__shared__ uint64_t C[25];
	__shared__ uint64_t D[25];

	A[t] = input[t];

	uint32_t* nonce_pos = (uint32_t*)(A) + 9;
	nonce += g;
	nonce_pos[0] = (nonce_pos[0] & 0xFFFFFFU) | ((nonce & 0xFF) << 24);
	nonce_pos[1] = (nonce_pos[1] & 0xFF000000U) | (nonce >> 8);

	uint32_t wordIndex = input_size / sizeof(uint64_t);
	const uint32_t tail_size = input_size % sizeof(uint64_t);

	A[wordIndex] ^= (uint64_t)(((uint64_t)(0x02 | (1 << 2))) << (tail_size * 8));
	A[16] ^= 0x8000000000000000ULL;

	#pragma unroll(ROUNDS)
	for (int i = 0; i < ROUNDS; ++i)
	{
		C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
		D[t] = C[b[20 + s]] ^ R64(C[b[5 + s]], 1, 63);
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

} // AstroBWT_Dero
