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

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void fillAes_name(__global void* state, __global void* out, uint batch_size, uint rx_version)
{
	__local uint T[2048];

	const uint global_index = get_global_id(0);
	if (global_index >= batch_size * 4)
		return;

	const uint idx = global_index / 4;
	const uint sub = global_index % 4;

	for (uint i = get_local_id(0), step = get_local_size(0); i < 2048; i += step)
		T[i] = AES_TABLE[i];

	barrier(CLK_LOCAL_MEM_FENCE);

#if num_rounds != 4
	const uint k[4] = { AES_KEY_FILL[sub * 4], AES_KEY_FILL[sub * 4 + 1], AES_KEY_FILL[sub * 4 + 2], AES_KEY_FILL[sub * 4 + 3] };
#else
	const bool b1 = (rx_version < 104);
	const bool b2 = (sub < 2);

	uint k[16];
	k[ 0] = b1 ? 0xf890465du : (b2 ? 0x6421aaddu : 0xb5826f73u);
	k[ 1] = b1 ? 0x7ffbe4a6u : (b2 ? 0xd1833ddbu : 0xe3d6a7a6u);
	k[ 2] = b1 ? 0x141f82b7u : (b2 ? 0x2f546d2bu : 0x3d518b6du);
	k[ 3] = b1 ? 0xcf359e95u : (b2 ? 0x99e5d23fu : 0x229effb4u);
	k[ 4] = b1 ? 0x6a55c450u : (b2 ? 0xb20e3450u : 0xc7566bf3u);
	k[ 5] = b1 ? 0xfee8278au : (b2 ? 0xb6913f55u : 0x9c10b3d9u);
	k[ 6] = b1 ? 0xbd5c5ac3u : (b2 ? 0x06f79d53u : 0xe9024d4eu);
	k[ 7] = b1 ? 0x6741ffdcu : (b2 ? 0xa5dfcde5u : 0xb272b7d2u);
	k[ 8] = b1 ? 0x114c47a4u : (b2 ? 0x5c3ed904u : 0xf273c9e7u);
	k[ 9] = b1 ? 0xd524fde4u : (b2 ? 0x515e7bafu : 0xf765a38bu);
	k[10] = b1 ? 0xa7279ad2u : (b2 ? 0x0aa4679fu : 0x2ba9660au);
	k[11] = b1 ? 0x3d324aacu : (b2 ? 0x171c02bfu : 0xf63befa7u);
	k[12] = b1 ? 0x810c3a2au : (b2 ? 0x85623763u : 0x7a7cd609u);
	k[13] = b1 ? 0x99a9aeffu : (b2 ? 0xe78f5d08u : 0x915839deu);
	k[14] = b1 ? 0x42d3dbd9u : (b2 ? 0xcd673785u : 0x0c06d1fdu);
	k[15] = b1 ? 0x76f6db08u : (b2 ? 0xd8ded291u : 0xc0b0762du);
#endif

	__global uint* s = ((__global uint*) state) + idx * (64 / sizeof(uint)) + sub * (16 / sizeof(uint));
	uint x[4] = { s[0], s[1], s[2], s[3] };

	const uint s1 = (sub & 1) ? 8 : 24;
	const uint s3 = (sub & 1) ? 24 : 8;

	__global uint4* p = ((__global uint4*) out) + idx * (outputSize0 / sizeof(uint4)) + sub;

	const __local uint* const t0 = (sub & 1) ? T : (T + 1024);
	const __local uint* const t1 = (sub & 1) ? (T + 256) : (T + 1792);
	const __local uint* const t2 = (sub & 1) ? (T + 512) : (T + 1536);
	const __local uint* const t3 = (sub & 1) ? (T + 768) : (T + 1280);

	#pragma unroll unroll_factor
	for (uint i = 0; i < outputSize / sizeof(uint4); i += 4, p += 4)
	{
		uint y[4];

#if num_rounds != 4
		y[0] = t0[get_byte32(x[0], 0)] ^ t1[get_byte32(x[1], s1)] ^ t2[get_byte32(x[2], 16)] ^ t3[get_byte32(x[3], s3)] ^ k[0];
		y[1] = t0[get_byte32(x[1], 0)] ^ t1[get_byte32(x[2], s1)] ^ t2[get_byte32(x[3], 16)] ^ t3[get_byte32(x[0], s3)] ^ k[1];
		y[2] = t0[get_byte32(x[2], 0)] ^ t1[get_byte32(x[3], s1)] ^ t2[get_byte32(x[0], 16)] ^ t3[get_byte32(x[1], s3)] ^ k[2];
		y[3] = t0[get_byte32(x[3], 0)] ^ t1[get_byte32(x[0], s1)] ^ t2[get_byte32(x[1], 16)] ^ t3[get_byte32(x[2], s3)] ^ k[3];

		*p = *(uint4*)(y);

		x[0] = y[0];
		x[1] = y[1];
		x[2] = y[2];
		x[3] = y[3];
#else
		y[0] = t0[get_byte32(x[0], 0)] ^ t1[get_byte32(x[1], s1)] ^ t2[get_byte32(x[2], 16)] ^ t3[get_byte32(x[3], s3)] ^ k[ 0];
		y[1] = t0[get_byte32(x[1], 0)] ^ t1[get_byte32(x[2], s1)] ^ t2[get_byte32(x[3], 16)] ^ t3[get_byte32(x[0], s3)] ^ k[ 1];
		y[2] = t0[get_byte32(x[2], 0)] ^ t1[get_byte32(x[3], s1)] ^ t2[get_byte32(x[0], 16)] ^ t3[get_byte32(x[1], s3)] ^ k[ 2];
		y[3] = t0[get_byte32(x[3], 0)] ^ t1[get_byte32(x[0], s1)] ^ t2[get_byte32(x[1], 16)] ^ t3[get_byte32(x[2], s3)] ^ k[ 3];

		x[0] = t0[get_byte32(y[0], 0)] ^ t1[get_byte32(y[1], s1)] ^ t2[get_byte32(y[2], 16)] ^ t3[get_byte32(y[3], s3)] ^ k[ 4];
		x[1] = t0[get_byte32(y[1], 0)] ^ t1[get_byte32(y[2], s1)] ^ t2[get_byte32(y[3], 16)] ^ t3[get_byte32(y[0], s3)] ^ k[ 5];
		x[2] = t0[get_byte32(y[2], 0)] ^ t1[get_byte32(y[3], s1)] ^ t2[get_byte32(y[0], 16)] ^ t3[get_byte32(y[1], s3)] ^ k[ 6];
		x[3] = t0[get_byte32(y[3], 0)] ^ t1[get_byte32(y[0], s1)] ^ t2[get_byte32(y[1], 16)] ^ t3[get_byte32(y[2], s3)] ^ k[ 7];

		y[0] = t0[get_byte32(x[0], 0)] ^ t1[get_byte32(x[1], s1)] ^ t2[get_byte32(x[2], 16)] ^ t3[get_byte32(x[3], s3)] ^ k[ 8];
		y[1] = t0[get_byte32(x[1], 0)] ^ t1[get_byte32(x[2], s1)] ^ t2[get_byte32(x[3], 16)] ^ t3[get_byte32(x[0], s3)] ^ k[ 9];
		y[2] = t0[get_byte32(x[2], 0)] ^ t1[get_byte32(x[3], s1)] ^ t2[get_byte32(x[0], 16)] ^ t3[get_byte32(x[1], s3)] ^ k[10];
		y[3] = t0[get_byte32(x[3], 0)] ^ t1[get_byte32(x[0], s1)] ^ t2[get_byte32(x[1], 16)] ^ t3[get_byte32(x[2], s3)] ^ k[11];

		x[0] = t0[get_byte32(y[0], 0)] ^ t1[get_byte32(y[1], s1)] ^ t2[get_byte32(y[2], 16)] ^ t3[get_byte32(y[3], s3)] ^ k[12];
		x[1] = t0[get_byte32(y[1], 0)] ^ t1[get_byte32(y[2], s1)] ^ t2[get_byte32(y[3], 16)] ^ t3[get_byte32(y[0], s3)] ^ k[13];
		x[2] = t0[get_byte32(y[2], 0)] ^ t1[get_byte32(y[3], s1)] ^ t2[get_byte32(y[0], 16)] ^ t3[get_byte32(y[1], s3)] ^ k[14];
		x[3] = t0[get_byte32(y[3], 0)] ^ t1[get_byte32(y[0], s1)] ^ t2[get_byte32(y[1], 16)] ^ t3[get_byte32(y[2], s3)] ^ k[15];

		*p = *(uint4*)(x);
#endif
	}

	*(__global uint4*)(s) = *(uint4*)(x);
}
