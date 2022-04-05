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

#define BLOCK_SIZE 1024
#define DATA_SIZE 9973
#define DATA_STRIDE 10240
#define BITS 14
#define COUNTERS_SIZE (1 << BITS)

inline uint16_t atomic_inc16(__local uint16_t* value)
{
	const size_t k = (size_t) value;
	if ((k & 3) == 0) {
		return atomic_add((__local uint32_t*) value, 1);
	}
	return atomic_add((__local uint32_t*)(k - 2), 0x10000) >> 16;
}

__attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
__kernel void BWT_preprocess(__global const uint8_t* datas, __global uint32_t* keys)
{
	const uint32_t data_offset = get_group_id(0) * DATA_STRIDE;
	const uint32_t tid = get_local_id(0);

	__local uint32_t counters_buf[COUNTERS_SIZE / 2];
	__local uint16_t* counters = (__local uint16_t*) counters_buf;
	for (uint32_t i = tid; i < COUNTERS_SIZE / 2; i += BLOCK_SIZE) {
		counters_buf[i] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	datas += data_offset;
	keys += data_offset;

	for (uint32_t i = tid; i < DATA_SIZE; i += BLOCK_SIZE) {
		const uint32_t k0 = datas[i];
		const uint32_t k1 = datas[i + 1];
		const uint32_t k = ((k0 << 8) | k1) >> (16 - BITS);
		atomic_inc16(counters + k);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	#pragma unroll BITS
	for (int k = 0; k < BITS; ++k) {
		for (uint32_t t1 = tid; t1 < ((COUNTERS_SIZE / 2) >> k); t1 += BLOCK_SIZE) {
			const uint32_t i = (t1 << (k + 1)) + ((1 << (k + 1)) - 1);
			counters[i] += counters[i - (1 << k)];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tid == 0) {
		counters[COUNTERS_SIZE - 1] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	#pragma unroll BITS
	for (int k = BITS - 1; k >= 0; --k) {
		for (uint32_t t1 = tid; t1 < ((COUNTERS_SIZE / 2) >> k); t1 += BLOCK_SIZE) {
			const uint32_t i = (t1 << (k + 1)) + ((1 << (k + 1)) - 1);
			const uint16_t old = counters[i];
			counters[i] = old + counters[i - (1 << k)];
			counters[i - (1 << k)] = old;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (uint32_t i = tid; i < DATA_SIZE; i += BLOCK_SIZE) {
		const uint32_t k0 = datas[i];
		const uint32_t k1 = datas[i + 1];
		const uint32_t k = (k0 << 8) | k1;
		const uint32_t index = atomic_inc16(counters + (k >> (16 - BITS)));
		keys[index] = (k << 16) | i;
	}
}

inline void fix_order(__global const uint8_t* input, uint32_t a, uint32_t b, __global uint32_t* keys)
{
	const uint32_t ka = keys[a];
	const uint32_t kb = keys[b];
	const uint32_t index_a = ka & 0xFFFF;
	const uint32_t index_b = kb & 0xFFFF;

	const uint32_t value_a =
		(((uint32_t)input[index_a + 1]) << 24) |
		(((uint32_t)input[index_a + 2]) << 16) |
		(((uint32_t)input[index_a + 3]) << 8) |
		((uint32_t)input[index_a + 4]);

	const uint32_t value_b =
		(((uint32_t)input[index_b + 1]) << 24) |
		(((uint32_t)input[index_b + 2]) << 16) |
		(((uint32_t)input[index_b + 3]) << 8) |
		((uint32_t)input[index_b + 4]);

	if (value_a > value_b)
	{
		keys[a] = kb;
		keys[b] = ka;
	}
}

__attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
__kernel void BWT_fix_order(__global const uint8_t* datas, __global uint32_t* keys, __global uint16_t* values)
{
	const uint32_t tid = get_local_id(0);
	const uint32_t gid = get_group_id(0);

	const uint32_t data_offset = gid * 10240;

	const uint32_t N = 9973;

	datas += data_offset;
	keys += data_offset;
	values += data_offset;

	for (uint32_t i = tid, N1 = N - 1; i < N1; i += BLOCK_SIZE)
	{
		const uint32_t value = keys[i] >> (32 - BITS);
		if (value == (keys[i + 1] >> (32 - BITS)))
		{
			if (i && (value == (keys[i - 1] >> (32 - BITS))))
				continue;

			uint32_t n = i + 2;
			while ((n < N) && (value == (keys[n] >> (32 - BITS))))
				++n;

			for (uint32_t j = i; j < n; ++j)
				for (uint32_t k = j + 1; k < n; ++k)
					fix_order(datas, j, k, keys);
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	for (uint32_t i = tid; i < N; i += BLOCK_SIZE) {
		values[i] = keys[i];
	}
}

__kernel void find_shares(__global const uint64_t* hashes, uint64_t target, __global uint32_t* shares)
{
	const uint32_t global_index = get_global_id(0);

	if (hashes[global_index * 4 + 3] >= target) {
		return;
	}

	const uint32_t idx = atomic_inc(shares + 0xFF);
	if (idx < 0xFF)
		shares[idx] = global_index;
}

#undef BLOCK_SIZE
#undef DATA_SIZE
#undef DATA_STRIDE
#undef BITS
#undef COUNTERS_SIZE
