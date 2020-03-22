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

#define ROTATE(v,c) (rotate(v,(uint32_t)c))
#define XOR(v,w) ((v) ^ (w))
#define PLUS(v,w) ((v) + (w))

__attribute__((reqd_work_group_size(SALSA20_GROUP_SIZE, 1, 1)))
__kernel void Salsa20_XORKeyStream(__global const uint32_t* keys, __global uint32_t* outputs, __global uint32_t* output_sizes, uint32_t output_stride)
{
	const uint32_t t = get_local_id(0);
	const uint32_t g = get_group_id(0);

	// Put zeroes in the beginning
	const uint64_t output_offset = g * ((uint64_t)output_stride) + 128;
	{
		__global uint32_t* p = outputs + (output_offset - 128) / sizeof(uint32_t);
		for (uint32_t i = t; i < 128 / sizeof(uint32_t); i += SALSA20_GROUP_SIZE)
			p[i] = 0;
	}
	
	__global const uint32_t* k = keys + g * 8;
	__global uint32_t* output = outputs + (output_offset + (t * 64)) / sizeof(uint32_t);
	const uint32_t output_size = output_sizes[g];

	const uint32_t j1 = k[0];
	const uint32_t j2 = k[1];
	const uint32_t j3 = k[2];
	const uint32_t j4 = k[3];
	const uint32_t j11 = k[4];
	const uint32_t j12 = k[5];
	const uint32_t j13 = k[6];
	const uint32_t j14 = k[7];
	const uint32_t j0 = 0x61707865U;
	const uint32_t j5 = 0x3320646EU;
	const uint32_t j10 = 0x79622D32U;
	const uint32_t j15 = 0x6B206574U;
	const uint32_t j6 = 0;
	const uint32_t j7 = 0;
	const uint32_t j8 = 0;
	const uint32_t j9 = 0;

	for (uint32_t i = t * 64; i < output_size; i += SALSA20_GROUP_SIZE * 64)
	{
		const uint32_t j8_1 = j8 + (i / 64);

		uint32_t x0 = j0;
		uint32_t x1 = j1;
		uint32_t x2 = j2;
		uint32_t x3 = j3;
		uint32_t x4 = j4;
		uint32_t x5 = j5;
		uint32_t x6 = j6;
		uint32_t x7 = j7;
		uint32_t x8 = j8_1;
		uint32_t x9 = j9;
		uint32_t x10 = j10;
		uint32_t x11 = j11;
		uint32_t x12 = j12;
		uint32_t x13 = j13;
		uint32_t x14 = j14;
		uint32_t x15 = j15;

		#pragma unroll(5)
		for (uint32_t j = 0; j < 10; ++j)
		{
			x4  = XOR( x4,ROTATE(PLUS( x0,x12), 7));
			x8  = XOR( x8,ROTATE(PLUS( x4, x0), 9));
			x12 = XOR(x12,ROTATE(PLUS( x8, x4),13));
			x0  = XOR( x0,ROTATE(PLUS(x12, x8),18));
			x9  = XOR( x9,ROTATE(PLUS( x5, x1), 7));
			x13 = XOR(x13,ROTATE(PLUS( x9, x5), 9));
			x1  = XOR( x1,ROTATE(PLUS(x13, x9),13));
			x5  = XOR( x5,ROTATE(PLUS( x1,x13),18));
			x14 = XOR(x14,ROTATE(PLUS(x10, x6), 7));
			x2  = XOR( x2,ROTATE(PLUS(x14,x10), 9));
			x6  = XOR( x6,ROTATE(PLUS( x2,x14),13));
			x10 = XOR(x10,ROTATE(PLUS( x6, x2),18));
			x3  = XOR( x3,ROTATE(PLUS(x15,x11), 7));
			x7  = XOR( x7,ROTATE(PLUS( x3,x15), 9));
			x11 = XOR(x11,ROTATE(PLUS( x7, x3),13));
			x15 = XOR(x15,ROTATE(PLUS(x11, x7),18));
			x1  = XOR( x1,ROTATE(PLUS( x0, x3), 7));
			x2  = XOR( x2,ROTATE(PLUS( x1, x0), 9));
			x3  = XOR( x3,ROTATE(PLUS( x2, x1),13));
			x0  = XOR( x0,ROTATE(PLUS( x3, x2),18));
			x6  = XOR( x6,ROTATE(PLUS( x5, x4), 7));
			x7  = XOR( x7,ROTATE(PLUS( x6, x5), 9));
			x4  = XOR( x4,ROTATE(PLUS( x7, x6),13));
			x5  = XOR( x5,ROTATE(PLUS( x4, x7),18));
			x11 = XOR(x11,ROTATE(PLUS(x10, x9), 7));
			x8  = XOR( x8,ROTATE(PLUS(x11,x10), 9));
			x9  = XOR( x9,ROTATE(PLUS( x8,x11),13));
			x10 = XOR(x10,ROTATE(PLUS( x9, x8),18));
			x12 = XOR(x12,ROTATE(PLUS(x15,x14), 7));
			x13 = XOR(x13,ROTATE(PLUS(x12,x15), 9));
			x14 = XOR(x14,ROTATE(PLUS(x13,x12),13));
			x15 = XOR(x15,ROTATE(PLUS(x14,x13),18));
		}

		output[0]  = PLUS(x0, j0);
		output[1]  = PLUS(x1, j1);
		output[2]  = PLUS(x2, j2);
		output[3]  = PLUS(x3, j3);
		output[4]  = PLUS(x4, j4);
		output[5]  = PLUS(x5, j5);
		output[6]  = PLUS(x6, j6);
		output[7]  = PLUS(x7, j7);
		output[8]  = PLUS(x8, j8_1);
		output[9]  = PLUS(x9, j9);
		output[10] = PLUS(x10,j10);
		output[11] = PLUS(x11,j11);
		output[12] = PLUS(x12,j12);
		output[13] = PLUS(x13,j13);
		output[14] = PLUS(x14,j14);
		output[15] = PLUS(x15,j15);

		output += (SALSA20_GROUP_SIZE * 64) / sizeof(uint32_t);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	// Put zeroes after output's end
	if (t < 16)
	{
		__global uint32_t* p = outputs + (output_offset + output_size + 3) / sizeof(uint32_t);
		p[t] = 0;
	}

	if ((t == 0) && (output_size & 3))
		outputs[(output_offset + output_size) / sizeof(uint32_t)] &= 0xFFFFFFFFU >> ((4 - (output_size & 3)) << 3);
}
