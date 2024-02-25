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

__constant static const uchar blake2b_sigma[12 * 16] = {
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
	14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
	11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4,
	7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8,
	9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13,
	2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9,
	12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11,
	13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
	6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
	10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
	14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3,
};

enum Blake2b_IV
{
	iv0 = 0x6a09e667f3bcc908ul,
	iv1 = 0xbb67ae8584caa73bul,
	iv2 = 0x3c6ef372fe94f82bul,
	iv3 = 0xa54ff53a5f1d36f1ul,
	iv4 = 0x510e527fade682d1ul,
	iv5 = 0x9b05688c2b3e6c1ful,
	iv6 = 0x1f83d9abfb41bd6bul,
	iv7 = 0x5be0cd19137e2179ul,
};

ulong rotr64(ulong a, ulong shift) { return rotate(a, 64 - shift); }

#define G(r, i, a, b, c, d)                                                    \
	do {                                                                       \
		a = a + b + m[blake2b_sigma[r * 16 + 2 * i + 0]];                      \
		d = rotr64(d ^ a, 32);                                                 \
		c = c + d;                                                             \
		b = rotr64(b ^ c, 24);                                                 \
		a = a + b + m[blake2b_sigma[r * 16 + 2 * i + 1]];                      \
		d = rotr64(d ^ a, 16);                                                 \
		c = c + d;                                                             \
		b = rotr64(b ^ c, 63);                                                 \
	} while (0)

#define ROUND(r)                                                               \
	do {                                                                       \
		G(r, 0, v[0], v[4], v[8], v[12]);                                      \
		G(r, 1, v[1], v[5], v[9], v[13]);                                      \
		G(r, 2, v[2], v[6], v[10], v[14]);                                     \
		G(r, 3, v[3], v[7], v[11], v[15]);                                     \
		G(r, 4, v[0], v[5], v[10], v[15]);                                     \
		G(r, 5, v[1], v[6], v[11], v[12]);                                     \
		G(r, 6, v[2], v[7], v[8], v[13]);                                      \
		G(r, 7, v[3], v[4], v[9], v[14]);                                      \
	} while (0)

#define BLAKE2B_ROUNDS() ROUND(0);ROUND(1);ROUND(2);ROUND(3);ROUND(4);ROUND(5);ROUND(6);ROUND(7);ROUND(8);ROUND(9);ROUND(10);ROUND(11);

void blake2b_512_process_single_block(ulong *h, const ulong* m, uint blockTemplateSize)
{
	ulong v[16] =
	{
		iv0 ^ 0x01010040, iv1, iv2, iv3, iv4                    , iv5,  iv6, iv7,
		iv0             , iv1, iv2, iv3, iv4 ^ blockTemplateSize, iv5, ~iv6, iv7,
	};

	BLAKE2B_ROUNDS();

	h[0] = v[0] ^ v[ 8] ^ iv0 ^ 0x01010040;
	h[1] = v[1] ^ v[ 9] ^ iv1;
	h[2] = v[2] ^ v[10] ^ iv2;
	h[3] = v[3] ^ v[11] ^ iv3;
	h[4] = v[4] ^ v[12] ^ iv4;
	h[5] = v[5] ^ v[13] ^ iv5;
	h[6] = v[6] ^ v[14] ^ iv6;
	h[7] = v[7] ^ v[15] ^ iv7;
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void blake2b_initial_hash(__global void *out, __global const void* blockTemplate, uint blockTemplateSize, uint start_nonce)
{
	const uint global_index = get_global_id(0);

	__global const ulong* p = (__global const ulong*) blockTemplate;
	ulong m[16] = {
		(blockTemplateSize >   0) ? p[ 0] : 0,
		(blockTemplateSize >   8) ? p[ 1] : 0,
		(blockTemplateSize >  16) ? p[ 2] : 0,
		(blockTemplateSize >  24) ? p[ 3] : 0,
		(blockTemplateSize >  32) ? p[ 4] : 0,
		(blockTemplateSize >  40) ? p[ 5] : 0,
		(blockTemplateSize >  48) ? p[ 6] : 0,
		(blockTemplateSize >  56) ? p[ 7] : 0,
		(blockTemplateSize >  64) ? p[ 8] : 0,
		(blockTemplateSize >  72) ? p[ 9] : 0,
		(blockTemplateSize >  80) ? p[10] : 0,
		(blockTemplateSize >  88) ? p[11] : 0,
		(blockTemplateSize >  96) ? p[12] : 0,
		(blockTemplateSize > 104) ? p[13] : 0,
		(blockTemplateSize > 112) ? p[14] : 0,
		(blockTemplateSize > 120) ? p[15] : 0,
	};

	if (blockTemplateSize % sizeof(ulong))
		m[blockTemplateSize / sizeof(ulong)] &= (ulong)(-1) >> (64 - (blockTemplateSize % sizeof(ulong)) * 8);

	const ulong nonce = start_nonce + global_index;
	m[4] = (m[4] & ((ulong)(-1) >>  8)) | (nonce << 56);
	m[5] = (m[5] & ((ulong)(-1) << 24)) | (nonce >>  8);

	ulong hash[8];
	blake2b_512_process_single_block(hash, m, blockTemplateSize);

	__global ulong* t = ((__global ulong*) out) + global_index * 8;
	t[0] = hash[0];
	t[1] = hash[1];
	t[2] = hash[2];
	t[3] = hash[3];
	t[4] = hash[4];
	t[5] = hash[5];
	t[6] = hash[6];
	t[7] = hash[7];
}

void blake2b_512_process_double_block_variable(ulong *out, ulong* m, __global const ulong* in, uint in_len, uint out_len)
{
	ulong v[16] =
	{
		iv0 ^ (0x01010000u | out_len), iv1, iv2, iv3, iv4      , iv5, iv6, iv7,
		iv0               , iv1, iv2, iv3, iv4 ^ 128, iv5, iv6, iv7,
	};

	BLAKE2B_ROUNDS();

	ulong h[8];
	v[0] = h[0] = v[0] ^ v[8] ^ iv0 ^ (0x01010000u | out_len);
	v[1] = h[1] = v[1] ^ v[9] ^ iv1;
	v[2] = h[2] = v[2] ^ v[10] ^ iv2;
	v[3] = h[3] = v[3] ^ v[11] ^ iv3;
	v[4] = h[4] = v[4] ^ v[12] ^ iv4;
	v[5] = h[5] = v[5] ^ v[13] ^ iv5;
	v[6] = h[6] = v[6] ^ v[14] ^ iv6;
	v[7] = h[7] = v[7] ^ v[15] ^ iv7;
	v[8] = iv0;
	v[9] = iv1;
	v[10] = iv2;
	v[11] = iv3;
	v[12] = iv4 ^ in_len;
	v[13] = iv5;
	v[14] = ~iv6;
	v[15] = iv7;

	m[ 0] = (in_len > 128) ? in[16] : 0;
	m[ 1] = (in_len > 136) ? in[17] : 0;
	m[ 2] = (in_len > 144) ? in[18] : 0;
	m[ 3] = (in_len > 152) ? in[19] : 0;
	m[ 4] = (in_len > 160) ? in[20] : 0;
	m[ 5] = (in_len > 168) ? in[21] : 0;
	m[ 6] = (in_len > 176) ? in[22] : 0;
	m[ 7] = (in_len > 184) ? in[23] : 0;
	m[ 8] = (in_len > 192) ? in[24] : 0;
	m[ 9] = (in_len > 200) ? in[25] : 0;
	m[10] = (in_len > 208) ? in[26] : 0;
	m[11] = (in_len > 216) ? in[27] : 0;
	m[12] = (in_len > 224) ? in[28] : 0;
	m[13] = (in_len > 232) ? in[29] : 0;
	m[14] = (in_len > 240) ? in[30] : 0;
	m[15] = (in_len > 248) ? in[31] : 0;

	if (in_len % sizeof(ulong))
		m[(in_len - 128) / sizeof(ulong)] &= (ulong)(-1) >> (64 - (in_len % sizeof(ulong)) * 8);

	BLAKE2B_ROUNDS();

	if (out_len >  0) out[0] = h[0] ^ v[0] ^ v[8];
	if (out_len >  8) out[1] = h[1] ^ v[1] ^ v[9];
	if (out_len > 16) out[2] = h[2] ^ v[2] ^ v[10];
	if (out_len > 24) out[3] = h[3] ^ v[3] ^ v[11];
	if (out_len > 32) out[4] = h[4] ^ v[4] ^ v[12];
	if (out_len > 40) out[5] = h[5] ^ v[5] ^ v[13];
	if (out_len > 48) out[6] = h[6] ^ v[6] ^ v[14];
	if (out_len > 56) out[7] = h[7] ^ v[7] ^ v[15];
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void blake2b_initial_hash_double(__global void *out, __global const void* blockTemplate, uint blockTemplateSize, uint start_nonce)
{
	const uint global_index = get_global_id(0);

	__global const ulong* p = (__global const ulong*) blockTemplate;

	ulong m[16] = { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15] };

	const ulong nonce = start_nonce + global_index;
	m[4] = (m[4] & ((ulong)(-1) >>  8)) | (nonce << 56);
	m[5] = (m[5] & ((ulong)(-1) << 24)) | (nonce >>  8);

	ulong hash[8];
	blake2b_512_process_double_block_variable(hash, m, p, blockTemplateSize, 64);

	__global ulong* t = ((__global ulong*) out) + global_index * 8;
	t[0] = hash[0];
	t[1] = hash[1];
	t[2] = hash[2];
	t[3] = hash[3];
	t[4] = hash[4];
	t[5] = hash[5];
	t[6] = hash[6];
	t[7] = hash[7];
}

#define in_len 256

#define out_len 32
#define blake2b_512_process_double_block_name blake2b_512_process_double_block_32
#define blake2b_hash_registers_name blake2b_hash_registers_32
	#include "blake2b_double_block.cl"
#undef blake2b_hash_registers_name
#undef blake2b_512_process_double_block_name
#undef out_len

#define out_len 64
#define blake2b_512_process_double_block_name blake2b_512_process_double_block_64
#define blake2b_hash_registers_name blake2b_hash_registers_64
	#include "blake2b_double_block.cl"
#undef blake2b_hash_registers_name
#undef blake2b_512_process_double_block_name
#undef out_len
