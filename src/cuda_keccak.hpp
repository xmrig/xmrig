#ifdef __CUDACC__
__constant__
#else
const
#endif
uint64_t keccakf_rndc[24] ={
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

#if __CUDA_ARCH__ >= 350
	__forceinline__ __device__ uint64_t cuda_rotl64(const uint64_t value, const int offset)
	{
		uint2 result;
		if(offset >= 32)
		{
			asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
			asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		}
		else
		{
			asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
			asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		}
		return  __double_as_longlong(__hiloint2double(result.y, result.x));
	}
	#define rotl64_1(x, y) (cuda_rotl64((x), (y)))
#else
	#define rotl64_1(x, y) ((x) << (y) | ((x) >> (64 - (y))))
#endif

#define rotl64_2(x, y) rotl64_1(((x) >> 32) | ((x) << 32), (y))
#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

__device__ __forceinline__ void cn_keccakf2(uint64_t *s)
{
	uint8_t i;

	for(i = 0; i < 24; ++i)
	{
		uint64_t bc[5], tmpxor[5], tmp1, tmp2;

		tmpxor[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		tmpxor[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		tmpxor[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		tmpxor[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		tmpxor[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		bc[0] = tmpxor[0] ^ rotl64_1(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ rotl64_1(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ rotl64_1(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ rotl64_1(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ rotl64_1(tmpxor[1], 1);

		tmp1 = s[1] ^ bc[0];

		s[0] ^= bc[4];
		s[1] = rotl64_2(s[6] ^ bc[0], 12);
		s[6] = rotl64_1(s[9] ^ bc[3], 20);
		s[9] = rotl64_2(s[22] ^ bc[1], 29);
		s[22] = rotl64_2(s[14] ^ bc[3], 7);
		s[14] = rotl64_1(s[20] ^ bc[4], 18);
		s[20] = rotl64_2(s[2] ^ bc[1], 30);
		s[2] = rotl64_2(s[12] ^ bc[1], 11);
		s[12] = rotl64_1(s[13] ^ bc[2], 25);
		s[13] = rotl64_1(s[19] ^ bc[3], 8);
		s[19] = rotl64_2(s[23] ^ bc[2], 24);
		s[23] = rotl64_2(s[15] ^ bc[4], 9);
		s[15] = rotl64_1(s[4] ^ bc[3], 27);
		s[4] = rotl64_1(s[24] ^ bc[3], 14);
		s[24] = rotl64_1(s[21] ^ bc[0], 2);
		s[21] = rotl64_2(s[8] ^ bc[2], 23);
		s[8] = rotl64_2(s[16] ^ bc[0], 13);
		s[16] = rotl64_2(s[5] ^ bc[4], 4);
		s[5] = rotl64_1(s[3] ^ bc[2], 28);
		s[3] = rotl64_1(s[18] ^ bc[2], 21);
		s[18] = rotl64_1(s[17] ^ bc[1], 15);
		s[17] = rotl64_1(s[11] ^ bc[0], 10);
		s[11] = rotl64_1(s[7] ^ bc[1], 6);
		s[7] = rotl64_1(s[10] ^ bc[4], 3);
		s[10] = rotl64_1(tmp1, 1);

		tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
		tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
		tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
		tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
		tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
		s[0] ^= keccakf_rndc[i];
	}
}

__device__ __forceinline__ void cn_keccakf(uint64_t *s)
{
	uint64_t bc[5], tmpxor[5], tmp1, tmp2;

	for(int i = 0; i < 24; ++i)
	{
		tmpxor[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		tmpxor[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		tmpxor[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		tmpxor[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		tmpxor[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		bc[0] = tmpxor[0] ^ rotl64_1(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ rotl64_1(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ rotl64_1(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ rotl64_1(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ rotl64_1(tmpxor[1], 1);

		tmp1 = s[1] ^ bc[0];

		s[0] ^= bc[4];
		s[1] = rotl64_2(s[6] ^ bc[0], 12);
		s[6] = rotl64_1(s[9] ^ bc[3], 20);
		s[9] = rotl64_2(s[22] ^ bc[1], 29);
		s[22] = rotl64_2(s[14] ^ bc[3], 7);
		s[14] = rotl64_1(s[20] ^ bc[4], 18);
		s[20] = rotl64_2(s[2] ^ bc[1], 30);
		s[2] = rotl64_2(s[12] ^ bc[1], 11);
		s[12] = rotl64_1(s[13] ^ bc[2], 25);
		s[13] = rotl64_1(s[19] ^ bc[3], 8);
		s[19] = rotl64_2(s[23] ^ bc[2], 24);
		s[23] = rotl64_2(s[15] ^ bc[4], 9);
		s[15] = rotl64_1(s[4] ^ bc[3], 27);
		s[4] = rotl64_1(s[24] ^ bc[3], 14);
		s[24] = rotl64_1(s[21] ^ bc[0], 2);
		s[21] = rotl64_2(s[8] ^ bc[2], 23);
		s[8] = rotl64_2(s[16] ^ bc[0], 13);
		s[16] = rotl64_2(s[5] ^ bc[4], 4);
		s[5] = rotl64_1(s[3] ^ bc[2], 28);
		s[3] = rotl64_1(s[18] ^ bc[2], 21);
		s[18] = rotl64_1(s[17] ^ bc[1], 15);
		s[17] = rotl64_1(s[11] ^ bc[0], 10);
		s[11] = rotl64_1(s[7] ^ bc[1], 6);
		s[7] = rotl64_1(s[10] ^ bc[4], 3);
		s[10] = rotl64_1(tmp1, 1);

		tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
		tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
		tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
		tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
		tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
		s[0] ^= keccakf_rndc[i];
	}
}

__device__ __forceinline__ void cn_keccak(const uint64_t * __restrict__ input, int inlen, uint8_t * __restrict__ md)
{
	uint64_t st[25];

	#pragma unroll
	for (int i = 0; i < 25; ++i) {
		st[i] = 0;
	}

	// Input length must be a multiple of 136 and padded on the host side
	for (int i = 0; inlen > 0; i += 17, inlen -= 136) {
		#pragma unroll
		for (int j = 0; j < 17; ++j) {
			st[j] ^= input[i + j];
		}
		cn_keccakf(st);
	}

	MEMCPY8(md, st, 25);
	return;
}
