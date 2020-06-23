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

#define STAGE1_SIZE 147253

#define COUNTING_SORT_BITS 11
#define COUNTING_SORT_SIZE (1 << COUNTING_SORT_BITS)
#define FINAL_SORT_BATCH_SIZE COUNTING_SORT_SIZE
#define FINAL_SORT_OVERLAP_SIZE 32

__attribute__((reqd_work_group_size(BWT_GROUP_SIZE, 1, 1)))
__kernel void BWT(__global uint8_t* datas, __global uint32_t* data_sizes, uint32_t data_stride, __global uint64_t* indices, __global uint64_t* tmp_indices)
{
	const uint32_t tid = get_local_id(0);
	const uint32_t gid = get_group_id(0);

	__local int counters[COUNTING_SORT_SIZE][2];

	for (uint32_t i = tid; i < COUNTING_SORT_SIZE * 2; i += BWT_GROUP_SIZE)
		((__local int*)counters)[i] = 0;

	const uint64_t data_offset = (uint64_t)(gid) * data_stride;

	__global uint8_t* input = datas + data_offset + 128;
	const uint32_t N = data_sizes[gid] + 1;

	__global uint64_t* p = (__global uint64_t*)(input);
	volatile __local uint8_t* counters_atomic = (volatile __local uint8_t*)(counters);

	indices += data_offset;
	tmp_indices += data_offset;

	for (uint32_t i = tid; i < N; i += BWT_GROUP_SIZE)
	{
		const uint32_t index = i >> 3;
		const uint32_t bit_offset = (i & 7) << 3;

		const uint64_t a = p[index];
		uint64_t b = p[index + 1];
		if (bit_offset == 0)
			b = 0;

		uint64_t value = (a >> bit_offset) | (b << (64 - bit_offset));

		uint2 tmp;
		const uchar4 mask = (uchar4)(3, 2, 1, 0);

		tmp.x = as_uint(shuffle(as_uchar4(as_uint2(value).y), mask));
		tmp.y = as_uint(shuffle(as_uchar4(as_uint2(value).x), mask));

		value = as_ulong(tmp);

		indices[i] = (value & ((uint64_t)(-1) << 21)) | i;
		atomic_add((volatile __local int*)(counters_atomic + (((value >> (64 - COUNTING_SORT_BITS * 2)) & (COUNTING_SORT_SIZE - 1)) << 3)), 1);
		atomic_add((volatile __local int*)(counters_atomic + ((value >> (64 - COUNTING_SORT_BITS)) << 3) + 4), 1);
	}

	if (tid == 0)
	{
		int t0 = counters[0][0];
		int t1 = counters[0][1];
		counters[0][0] = t0 - 1;
		counters[0][1] = t1 - 1;
		for (uint32_t i = 1; i < COUNTING_SORT_SIZE; ++i)
		{
			t0 += counters[i][0];
			t1 += counters[i][1];
			counters[i][0] = t0 - 1;
			counters[i][1] = t1 - 1;
		}
	}

	for (int i = tid; i < N; i += BWT_GROUP_SIZE)
	{
		const uint64_t data = indices[i];
		const int k = atomic_sub((volatile __local int*)(counters_atomic + (((data >> (64 - COUNTING_SORT_BITS * 2)) & (COUNTING_SORT_SIZE - 1)) << 3)), 1);
		tmp_indices[k] = data;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = N - 1 - tid; i >= 0; i -= BWT_GROUP_SIZE)
	{
		const uint64_t data = tmp_indices[i];
		const int k = atomic_sub((volatile __local int*)(counters_atomic + ((data >> (64 - COUNTING_SORT_BITS)) << 3) + 4), 1);
		indices[k] = data;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	__local uint64_t* buf = (__local uint64_t*)(counters);
	for (uint32_t i = 0; i < N; i += FINAL_SORT_BATCH_SIZE - FINAL_SORT_OVERLAP_SIZE)
	{
		const uint32_t len = (N - i < FINAL_SORT_BATCH_SIZE) ? (N - i) : FINAL_SORT_BATCH_SIZE;

		for (uint32_t j = tid; j < len; j += BWT_GROUP_SIZE)
			buf[j] = indices[i + j];

		if (tid == 0)
		{
			uint64_t prev_t = buf[0];
			for (int i = 1; i < len; ++i)
			{
				uint64_t t = buf[i];
				if (t < prev_t)
				{
					const uint64_t t2 = prev_t;
					int j = i - 1;
					do
					{
						buf[j + 1] = prev_t;
						--j;
						if (j < 0)
							break;
						prev_t = buf[j];
					} while (t < prev_t);
					buf[j + 1] = t;
					t = t2;
				}
				prev_t = t;
			}
		}

		for (uint32_t j = tid; j < len; j += BWT_GROUP_SIZE)
			indices[i + j] = buf[j];
	}

	--input;
	__global uint8_t* output = (__global uint8_t*)(tmp_indices);
	for (int i = tid; i <= N; i += BWT_GROUP_SIZE)
		output[i] = input[indices[i] & ((1 << 21) - 1)];
}

__kernel void filter(uint32_t nonce, uint32_t bwt_max_size, __global const uint32_t* hashes, __global uint32_t* filtered_hashes)
{
	const uint32_t global_id = get_global_id(0);

	__global const uint32_t* hash = hashes + global_id * (32 / sizeof(uint32_t));
	const uint32_t stage2_size = STAGE1_SIZE + (*hash & 0xfffff);

	if (stage2_size < bwt_max_size)
	{
		const int index = atomic_add((volatile __global int*)(filtered_hashes), 1) * (36 / sizeof(uint32_t)) + 1;

		filtered_hashes[index] = nonce + global_id;

		#pragma unroll 8
		for (uint32_t i = 0; i < 8; ++i)
			filtered_hashes[index + i + 1] = hash[i];
	}
}

__kernel void prepare_batch2(__global uint32_t* hashes, __global uint32_t* filtered_hashes, __global uint32_t* data_sizes)
{
	const uint32_t global_id = get_global_id(0);
	const uint32_t N = filtered_hashes[0] - get_global_size(0);

	if (global_id == 0)
		filtered_hashes[0] = N;

	__global uint32_t* hash = hashes + global_id * 8;
	__global uint32_t* filtered_hash = filtered_hashes + (global_id + N) * 9 + 1;

	const uint32_t stage2_size = STAGE1_SIZE + (filtered_hash[1] & 0xfffff);
	data_sizes[global_id] = stage2_size;

	#pragma unroll 8
	for (uint32_t i = 0; i < 8; ++i)
		hash[i] = filtered_hash[i + 1];
}

__kernel void find_shares(__global const uint64_t* hashes, __global const uint32_t* filtered_hashes, uint64_t target, __global uint32_t* shares)
{
	const uint32_t global_index = get_global_id(0);

	if (hashes[global_index * 4 + 3] < target)
	{
		const uint32_t idx = atomic_inc(shares + 0xFF);
		if (idx < 0xFF)
			shares[idx] = filtered_hashes[(filtered_hashes[0] + global_index) * 9 + 1];
	}
}
