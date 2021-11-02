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

constexpr int STAGE1_SIZE = 147253;

__global__ __launch_bounds__(1024)
void BWT_preprocess(const uint8_t* datas, const uint32_t* data_sizes, uint32_t data_stride, uint32_t* keys_in, uint16_t* values_in, uint32_t* offsets_begin, uint32_t* offsets_end)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t gid = blockIdx.x;
	const uint32_t group_size = blockDim.x;

	const uint32_t data_offset = gid * data_stride;

	const uint8_t* input = datas + data_offset + 128;
	const uint32_t N = data_sizes[gid] + 1;

	keys_in += data_offset;
	values_in += data_offset;

	offsets_begin[gid] = data_offset;
	offsets_end[gid] = data_offset + N;

	for (uint32_t i = tid; i < N; i += group_size)
	{
		keys_in[i] =
			(static_cast<uint32_t>(input[i + 0]) << 24) |
			(static_cast<uint32_t>(input[i + 1]) << 16) |
			(static_cast<uint32_t>(input[i + 2]) << 8) |
			(i >> 16);

		values_in[i] = i & 0xFFFF;
	}
}

template<uint32_t begin_bit>
__device__ __forceinline__ void fix_order(const uint8_t* input, uint32_t a, uint32_t b, uint32_t* keys_in, uint16_t* values_in)
{
	constexpr uint32_t offset = (32 - begin_bit) / 8;

	const uint32_t key_in_a = keys_in[a];
	const uint32_t key_in_b = keys_in[b];

	const uint32_t value_in_a = values_in[a];
	const uint32_t value_in_b = values_in[b];

	const uint32_t index_a = (((key_in_a & 0xFF) << 16) | value_in_a) + offset;
	const uint32_t index_b = (((key_in_b & 0xFF) << 16) | value_in_b) + offset;

	const uint32_t value_a =
		(static_cast<uint32_t>(input[index_a + 0]) << 24) |
		(static_cast<uint32_t>(input[index_a + 1]) << 16) |
		(static_cast<uint32_t>(input[index_a + 2]) << 8) |
		static_cast<uint32_t>(input[index_a + 3]);

	const uint32_t value_b =
		(static_cast<uint32_t>(input[index_b + 0]) << 24) |
		(static_cast<uint32_t>(input[index_b + 1]) << 16) |
		(static_cast<uint32_t>(input[index_b + 2]) << 8) |
		static_cast<uint32_t>(input[index_b + 3]);

	if (value_a > value_b)
	{
		keys_in[a] = key_in_b;
		keys_in[b] = key_in_a;

		values_in[a] = value_in_b;
		values_in[b] = value_in_a;
	}
}

template<uint32_t begin_bit>
__global__ __launch_bounds__(1024)
void BWT_fix_order(const uint8_t* datas, const uint32_t* data_sizes, uint32_t data_stride, uint32_t* keys_in, uint16_t* values_in)
{
	const uint32_t tid = threadIdx.x;
	const uint32_t gid = blockIdx.x;
	const uint32_t group_size = blockDim.x;

	const uint32_t data_offset = gid * data_stride;
	const uint8_t* input = datas + data_offset + 128;

	const uint32_t N = data_sizes[gid] + 1;

	keys_in += data_offset;
	values_in += data_offset;

	for (uint32_t i = tid, N1 = N - 1; i < N1; i += group_size)
	{
		const uint32_t value = keys_in[i] >> begin_bit;
		if (value == (keys_in[i + 1] >> begin_bit))
		{
			if (i && (value == (keys_in[i - 1] >> begin_bit)))
				continue;

			uint32_t n = i + 2;
			while ((n < N) && (value == (keys_in[n] >> begin_bit)))
				++n;

			for (uint32_t j = i; j < n; ++j)
				for (uint32_t k = j + 1; k < n; ++k)
					fix_order<begin_bit>(input, j, k, keys_in, values_in);
		}
	}
}

template<uint32_t DATA_STRIDE>
__global__ __launch_bounds__(1024)
void BWT_apply(const uint8_t* datas, const uint32_t* data_sizes, uint32_t* keys_in, uint16_t* values_in, uint64_t* tmp_indices)
{
	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;

	const uint32_t data_index = global_index / DATA_STRIDE;
	const uint32_t data_offset = data_index * DATA_STRIDE;
	const uint32_t i = global_index - data_offset;

	const uint32_t N = data_sizes[data_index] + 1;

	if (i >= N)
		return;

	const uint8_t* input = datas + data_offset + 128 - 1;

	keys_in += data_offset;
	values_in += data_offset;
	tmp_indices += data_offset;

	uint8_t* output = (uint8_t*)(tmp_indices);

	output[i] = input[((keys_in[i] & 0xFF) << 16) | values_in[i]];
}

__global__ void __launch_bounds__(32) filter(uint32_t nonce, uint32_t bwt_max_size, const uint32_t* hashes, uint32_t* filtered_hashes)
{
	const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;

	const uint32_t* hash = hashes + global_id * (32 / sizeof(uint32_t));
	const uint32_t stage2_size = STAGE1_SIZE + (*hash & 0xfffff);

	if (stage2_size < bwt_max_size)
	{
		const int index = atomicAdd((int*)(filtered_hashes), 1) * (36 / sizeof(uint32_t)) + 1;

		filtered_hashes[index] = nonce + global_id;

		#pragma unroll(8)
		for (uint32_t i = 0; i < 8; ++i)
			filtered_hashes[index + i + 1] = hash[i];
	}
}

__global__ void __launch_bounds__(32) prepare_batch2(uint32_t* hashes, uint32_t* filtered_hashes, uint32_t* data_sizes)
{
	const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t N = filtered_hashes[0] - blockDim.x * gridDim.x;

	if (global_id == 0)
		filtered_hashes[0] = N;

	uint32_t* hash = hashes + global_id * 8;
	uint32_t* filtered_hash = filtered_hashes + (global_id + N) * 9 + 1;

	const uint32_t stage2_size = STAGE1_SIZE + (filtered_hash[1] & 0xfffff);
	data_sizes[global_id] = stage2_size;

	#pragma unroll(8)
	for (uint32_t i = 0; i < 8; ++i)
		hash[i] = filtered_hash[i + 1];
}

__global__ void __launch_bounds__(32) find_shares(const uint64_t* hashes, const uint32_t* filtered_hashes, uint64_t target, uint32_t* shares)
{
	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (hashes[global_index * 4 + 3] < target)
	{
		const int idx = atomicAdd((int*)(shares), 1) + 1;
		if (idx <= 10)
			shares[idx] = filtered_hashes[(filtered_hashes[0] + global_index) * 9 + 1];
	}
}

} // AstroBWT_Dero
