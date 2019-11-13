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

void blake2b_512_process_double_block_name(ulong *out, ulong* m, __global const ulong* in)
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
__kernel void blake2b_hash_registers_name(__global void *out, __global const void* in, uint inStrideBytes)
{
	const uint global_index = get_global_id(0);
	__global const ulong* p = ((__global const ulong*) in) + global_index * (inStrideBytes / sizeof(ulong));
	__global ulong* h = ((__global ulong*) out) + global_index * (out_len / sizeof(ulong));

	ulong m[16] = { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15] };

	ulong hash[8];
	blake2b_512_process_double_block_name(hash, m, p);

	if (out_len >  0) h[0] = hash[0];
	if (out_len >  8) h[1] = hash[1];
	if (out_len > 16) h[2] = hash[2];
	if (out_len > 24) h[3] = hash[3];
	if (out_len > 32) h[4] = hash[4];
	if (out_len > 40) h[5] = hash[5];
	if (out_len > 48) h[6] = hash[6];
	if (out_len > 56) h[7] = hash[7];
}
