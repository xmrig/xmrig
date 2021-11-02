#pragma once

/*
Copyright (c) 2019 SChernykh

This file is part of RandomX CUDA.

RandomX CUDA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX CUDA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX CUDA.  If not, see<http://www.gnu.org/licenses/>.
*/

static __constant__ const uint8_t blake2b_sigma[12 * 16] = {
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

struct Blake2b_IV
{
	enum Values : uint64_t
	{
		iv0 = 0x6a09e667f3bcc908ull,
		iv1 = 0xbb67ae8584caa73bull,
		iv2 = 0x3c6ef372fe94f82bull,
		iv3 = 0xa54ff53a5f1d36f1ull,
		iv4 = 0x510e527fade682d1ull,
		iv5 = 0x9b05688c2b3e6c1full,
		iv6 = 0x1f83d9abfb41bd6bull,
		iv7 = 0x5be0cd19137e2179ull,
	};
};

template<uint32_t shift> __device__ uint64_t rotr64(uint64_t a) { return (a >> shift) | (a << (64 - shift)); }

template<> __device__ uint64_t rotr64<63>(uint64_t a)
{
#if __CUDA_ARCH__ >= 320
	const uint32_t* data = (const uint32_t*) &a;
	const uint32_t result[2] = { __funnelshift_l(data[1], data[0], 1), __funnelshift_l(data[0], data[1], 1) };
	return *((const uint64_t*)result);
#else
    return (a >> 63) | (a << 1);
#endif
}

template<> __device__ uint64_t rotr64<32>(uint64_t a)
{
	const uint32_t* data = (const uint32_t*) &a;
	const uint32_t result[2] = { data[1], data[0] };
	return *((const uint64_t*) result);
}

template<uint32_t a, uint32_t b, uint32_t c, uint32_t d>
struct SelectBytes
{
	static_assert((a < 8) && (b < 8) && (c < 8) && (d < 8), "Byte index must be between 0 and 7");
	enum { value = a | (b << 4) | (c << 8) | (d << 12) };
};

template<> __device__ uint64_t rotr64<24>(uint64_t a)
{
#if __CUDA_ARCH__ >= 200
    const uint32_t* data = (const uint32_t*)&a;
	const uint32_t result[2] = { __byte_perm(data[0], data[1], SelectBytes<3, 4, 5, 6>::value), __byte_perm(data[0], data[1], SelectBytes<7, 0, 1, 2>::value) };
	return *((const uint64_t*) result);
#else
    return (a >> 24) | (a << 40);
#endif
}

template<> __device__ uint64_t rotr64<16>(uint64_t a)
{
#if __CUDA_ARCH__ >= 200
    const uint32_t* data = (const uint32_t*)&a;
	const uint32_t result[2] = { __byte_perm(data[0], data[1], SelectBytes<2, 3, 4, 5>::value), __byte_perm(data[0], data[1], SelectBytes<6, 7, 0, 1>::value) };
	return *((const uint64_t*)result);
#else
    return (a >> 16) | (a << 48);
#endif
}

#define G(r, i, a, b, c, d)                                                    \
	do {                                                                       \
		a = a + b + m[blake2b_sigma[r * 16 + 2 * i + 0]];                      \
		d = rotr64<32>(d ^ a);                                                 \
		c = c + d;                                                             \
		b = rotr64<24>(b ^ c);                                                 \
		a = a + b + m[blake2b_sigma[r * 16 + 2 * i + 1]];                      \
		d = rotr64<16>(d ^ a);                                                 \
		c = c + d;                                                             \
		b = rotr64<63>(b ^ c);                                                 \
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

__device__ void blake2b_512_process_single_block(uint64_t *h, const uint64_t* m, uint32_t in_len)
{
	uint64_t v[16] =
	{
		Blake2b_IV::iv0 ^ 0x01010040ul, Blake2b_IV::iv1, Blake2b_IV::iv2, Blake2b_IV::iv3, Blake2b_IV::iv4         , Blake2b_IV::iv5,  Blake2b_IV::iv6, Blake2b_IV::iv7,
		Blake2b_IV::iv0               , Blake2b_IV::iv1, Blake2b_IV::iv2, Blake2b_IV::iv3, Blake2b_IV::iv4 ^ in_len, Blake2b_IV::iv5, ~Blake2b_IV::iv6, Blake2b_IV::iv7,
	};

	BLAKE2B_ROUNDS();

	h[0] = v[0] ^ v[ 8] ^ Blake2b_IV::iv0 ^ 0x01010040ul;
	h[1] = v[1] ^ v[ 9] ^ Blake2b_IV::iv1;
	h[2] = v[2] ^ v[10] ^ Blake2b_IV::iv2;
	h[3] = v[3] ^ v[11] ^ Blake2b_IV::iv3;
	h[4] = v[4] ^ v[12] ^ Blake2b_IV::iv4;
	h[5] = v[5] ^ v[13] ^ Blake2b_IV::iv5;
	h[6] = v[6] ^ v[14] ^ Blake2b_IV::iv6;
	h[7] = v[7] ^ v[15] ^ Blake2b_IV::iv7;
}

template<uint32_t in_len> struct M_Mask { enum : uint64_t { value = uint64_t(-1) >> (64 - in_len * 8) }; };
template<> struct M_Mask<0> { enum : uint64_t { value = 0 }; };

template<uint32_t in_len, uint32_t out_len>
__device__ void blake2b_512_process_double_block(uint64_t *out, uint64_t* m, const uint64_t* in)
{
	static_assert(in_len > 128, "Double block must be larger than 128 bytes");
	static_assert(in_len <= 256, "Double block can't be larger than 256 bytes");

	uint64_t v[16] =
	{
		Blake2b_IV::iv0 ^ (0x01010000ul | out_len), Blake2b_IV::iv1, Blake2b_IV::iv2, Blake2b_IV::iv3, Blake2b_IV::iv4      , Blake2b_IV::iv5, Blake2b_IV::iv6, Blake2b_IV::iv7,
		Blake2b_IV::iv0               , Blake2b_IV::iv1, Blake2b_IV::iv2, Blake2b_IV::iv3, Blake2b_IV::iv4 ^ 128, Blake2b_IV::iv5, Blake2b_IV::iv6, Blake2b_IV::iv7,
	};

	BLAKE2B_ROUNDS();

	uint64_t h[8];
	v[0] = h[0] = v[0] ^ v[8] ^ Blake2b_IV::iv0 ^ (0x01010000ul | out_len);
	v[1] = h[1] = v[1] ^ v[9] ^ Blake2b_IV::iv1;
	v[2] = h[2] = v[2] ^ v[10] ^ Blake2b_IV::iv2;
	v[3] = h[3] = v[3] ^ v[11] ^ Blake2b_IV::iv3;
	v[4] = h[4] = v[4] ^ v[12] ^ Blake2b_IV::iv4;
	v[5] = h[5] = v[5] ^ v[13] ^ Blake2b_IV::iv5;
	v[6] = h[6] = v[6] ^ v[14] ^ Blake2b_IV::iv6;
	v[7] = h[7] = v[7] ^ v[15] ^ Blake2b_IV::iv7;
	v[8] = Blake2b_IV::iv0;
	v[9] = Blake2b_IV::iv1;
	v[10] = Blake2b_IV::iv2;
	v[11] = Blake2b_IV::iv3;
	v[12] = Blake2b_IV::iv4 ^ in_len;
	v[13] = Blake2b_IV::iv5;
	v[14] = ~Blake2b_IV::iv6;
	v[15] = Blake2b_IV::iv7;

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

	if (in_len % sizeof(uint64_t))
		m[(in_len - 128) / sizeof(uint64_t)] &= M_Mask<in_len % sizeof(uint64_t)>::value;

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

#undef G
#undef ROUND
#undef BLAKE2B_ROUNDS

__global__ void blake2b_initial_hash(void *out, const void* blockTemplate, uint32_t blockTemplate_len, uint32_t start_nonce)
{
	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;

	const uint64_t* p = (const uint64_t*) blockTemplate;
	uint64_t m[16] = {
		(blockTemplate_len > 0) ? p[0] : 0,
		(blockTemplate_len > 8) ? p[1] : 0,
		(blockTemplate_len > 16) ? p[2] : 0,
		(blockTemplate_len > 24) ? p[3] : 0,
		(blockTemplate_len > 32) ? p[4] : 0,
		(blockTemplate_len > 40) ? p[5] : 0,
		(blockTemplate_len > 48) ? p[6] : 0,
		(blockTemplate_len > 56) ? p[7] : 0,
		(blockTemplate_len > 64) ? p[8] : 0,
		(blockTemplate_len > 72) ? p[9] : 0,
		(blockTemplate_len > 80) ? p[10] : 0,
		(blockTemplate_len > 88) ? p[11] : 0,
		(blockTemplate_len > 96) ? p[12] : 0,
		(blockTemplate_len > 104) ? p[13] : 0,
		(blockTemplate_len > 112) ? p[14] : 0,
		(blockTemplate_len > 120) ? p[15] : 0,
	};

	if (blockTemplate_len % sizeof(uint64_t))
		m[blockTemplate_len / sizeof(uint64_t)] &= uint64_t(-1) >> (64 - (blockTemplate_len % sizeof(uint64_t)) * 8);

	const uint64_t nonce = start_nonce + global_index;
	m[4] = (m[4] & (uint64_t(-1) >>  8)) | (nonce << 56);
	m[5] = (m[5] & (uint64_t(-1) << 24)) | (nonce >>  8);

	uint64_t hash[8];
	blake2b_512_process_single_block(hash, m, blockTemplate_len);

	uint64_t* t = ((uint64_t*) out) + global_index * 8;
	t[0] = hash[0];
	t[1] = hash[1];
	t[2] = hash[2];
	t[3] = hash[3];
	t[4] = hash[4];
	t[5] = hash[5];
	t[6] = hash[6];
	t[7] = hash[7];
}

template<uint32_t registers_len, uint32_t registers_stride, uint32_t out_len>
__global__ void blake2b_hash_registers(void *out, const void* in)
{
	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t* p = ((const uint64_t*) in) + global_index * (registers_stride / sizeof(uint64_t));
	uint64_t* h = ((uint64_t*) out) + global_index * (out_len / sizeof(uint64_t));

	uint64_t m[16] = { p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15] };
	blake2b_512_process_double_block<registers_len, out_len>(h, m, p);
}
