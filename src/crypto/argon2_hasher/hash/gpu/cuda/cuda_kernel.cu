#include <driver_types.h>

#include <crypto/Argon2_constants.h>

#include "../../../common/common.h"

#include "crypto/argon2_hasher/hash/Hasher.h"
#include "crypto/argon2_hasher/hash/argon2/Argon2.h"

#include "CudaHasher.h"

#define THREADS_PER_LANE               32
#define BLOCK_SIZE_UINT4                64
#define BLOCK_SIZE_UINT                256
#define KERNEL_WORKGROUP_SIZE   		32
#define ARGON2_PREHASH_DIGEST_LENGTH_UINT   16
#define ARGON2_PREHASH_SEED_LENGTH_UINT     18


#include "blake2b.cu"

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if (__CUDA_ARCH__ >= 350)
    #define COMPUTE	\
        asm ("{"	\
            ".reg .u32 s1, s2, s3, s4;\n\t"	\
            "mul.lo.u32 s3, %0, %2;\n\t"	\
            "mul.hi.u32 s4, %0, %2;\n\t"	\
            "add.cc.u32 s3, s3, s3;\n\t"	\
            "addc.u32 s4, s4, s4;\n\t"	\
            "add.cc.u32 s1, %0, %2;\n\t"	\
            "addc.u32 s2, %1, %3;\n\t"	\
            "add.cc.u32 %0, s1, s3;\n\t"	\
            "addc.u32 %1, s2, s4;\n\t"	\
            "xor.b32 s1, %0, %6;\n\t"	\
            "xor.b32 %6, %1, %7;\n\t"	\
            "mov.b32 %7, s1;\n\t"	\
            "mul.lo.u32 s3, %4, %6;\n\t"	\
            "mul.hi.u32 s4, %4, %6;\n\t"	\
            "add.cc.u32 s3, s3, s3;\n\t"	\
            "addc.u32 s4, s4, s4;\n\t"	\
            "add.cc.u32 s1, %4, %6;\n\t"	\
            "addc.u32 s2, %5, %7;\n\t"	\
            "add.cc.u32 %4, s1, s3;\n\t"	\
            "addc.u32 %5, s2, s4;\n\t"	\
            "xor.b32 s3, %2, %4;\n\t"	\
            "xor.b32 s4, %3, %5;\n\t"	\
            "shf.r.wrap.b32 %3, s4, s3, 24;\n\t"	\
            "shf.r.wrap.b32 %2, s3, s4, 24;\n\t"	\
            "mul.lo.u32 s3, %0, %2;\n\t"	\
            "mul.hi.u32 s4, %0, %2;\n\t"	\
            "add.cc.u32 s3, s3, s3;\n\t"	\
            "addc.u32 s4, s4, s4;\n\t"	\
            "add.cc.u32 s1, %0, %2;\n\t"	\
            "addc.u32 s2, %1, %3;\n\t"	\
            "add.cc.u32 %0, s1, s3;\n\t"	\
            "addc.u32 %1, s2, s4;\n\t"	\
            "xor.b32 s3, %0, %6;\n\t"	\
            "xor.b32 s4, %1, %7;\n\t"	\
            "shf.r.wrap.b32 %7, s4, s3, 16;\n\t"	\
            "shf.r.wrap.b32 %6, s3, s4, 16;\n\t"	\
            "mul.lo.u32 s3, %4, %6;\n\t"	\
            "mul.hi.u32 s4, %4, %6;\n\t"	\
            "add.cc.u32 s3, s3, s3;\n\t"	\
            "addc.u32 s4, s4, s4;\n\t"	\
            "add.cc.u32 s1, %4, %6;\n\t"	\
            "addc.u32 s2, %5, %7;\n\t"	\
            "add.cc.u32 %4, s1, s3;\n\t"	\
            "addc.u32 %5, s2, s4;\n\t"	\
            "xor.b32 s3, %2, %4;\n\t"	\
            "xor.b32 s4, %3, %5;\n\t"	\
            "shf.r.wrap.b32 %3, s3, s4, 31;\n\t"	\
            "shf.r.wrap.b32 %2, s4, s3, 31;\n\t"	\
        "}" : "+r"(tmp_a.x), "+r"(tmp_a.y), "+r"(tmp_a.z), "+r"(tmp_a.w), "+r"(tmp_b.x), "+r"(tmp_b.y), "+r"(tmp_b.z), "+r"(tmp_b.w));
#else
    #define downsample(x, lo, hi) \
    { \
        lo = (uint32_t)x; \
        hi = (uint32_t)(x >> 32); \
    }

    #define upsample(lo, hi) (((uint64_t)(hi) << 32) | (uint64_t)(lo))

    #define rotate(x, n) (((x) >> (64-n)) | ((x) << n))

    #define fBlaMka(x, y) ((x) + (y) + 2 * upsample((uint32_t)(x) * (uint32_t)y, __umulhi((uint32_t)(x), (uint32_t)(y))))

    #define COMPUTE \
    { \
        uint64_t a64 = upsample(tmp_a.x, tmp_a.y); \
        uint64_t b64 = upsample(tmp_a.z, tmp_a.w); \
        uint64_t c64 = upsample(tmp_b.x, tmp_b.y); \
        uint64_t d64 = upsample(tmp_b.z, tmp_b.w); \
        a64 = fBlaMka(a64, b64);          \
        d64 = rotate(d64 ^ a64, 32);      \
        c64 = fBlaMka(c64, d64);          \
        b64 = rotate(b64 ^ c64, 40);      \
        a64 = fBlaMka(a64, b64);          \
        d64 = rotate(d64 ^ a64, 48);      \
        c64 = fBlaMka(c64, d64);          \
        b64 = rotate(b64 ^ c64, 1);       \
        downsample(a64, tmp_a.x, tmp_a.y); \
        downsample(b64, tmp_a.z, tmp_a.w); \
        downsample(c64, tmp_b.x, tmp_b.y); \
        downsample(d64, tmp_b.z, tmp_b.w); \
    }
#endif // __CUDA_ARCH__

#define G1(data)           \
{                           \
	COMPUTE \
	tmp_a.z = __shfl_sync(0xffffffff, tmp_a.z, i_shfl1_1); \
	tmp_a.w = __shfl_sync(0xffffffff, tmp_a.w, i_shfl1_1); \
	tmp_b.x = __shfl_sync(0xffffffff, tmp_b.x, i_shfl1_2); \
	tmp_b.y = __shfl_sync(0xffffffff, tmp_b.y, i_shfl1_2); \
	tmp_b.z = __shfl_sync(0xffffffff, tmp_b.z, i_shfl1_3); \
	tmp_b.w = __shfl_sync(0xffffffff, tmp_b.w, i_shfl1_3); \
}

#define G2(data)           \
{ \
	COMPUTE \
    data[i2_0_0] = tmp_a.x; \
    data[i2_0_1] = tmp_a.y; \
    data[i2_1_0] = tmp_a.z; \
    data[i2_1_1] = tmp_a.w; \
    data[i2_2_0] = tmp_b.x; \
    data[i2_2_1] = tmp_b.y; \
    data[i2_3_0] = tmp_b.z; \
    data[i2_3_1] = tmp_b.w; \
    __syncwarp(); \
}

#define G3(data)           \
{                           \
    tmp_a.x = data[i3_0_0]; \
    tmp_a.y = data[i3_0_1]; \
    tmp_a.z = data[i3_1_0]; \
    tmp_a.w = data[i3_1_1]; \
    tmp_b.x = data[i3_2_0]; \
    tmp_b.y = data[i3_2_1]; \
    tmp_b.z = data[i3_3_0]; \
    tmp_b.w = data[i3_3_1]; \
	COMPUTE \
	tmp_a.z = __shfl_sync(0xffffffff, tmp_a.z, i_shfl2_1); \
	tmp_a.w = __shfl_sync(0xffffffff, tmp_a.w, i_shfl2_1); \
	tmp_b.x = __shfl_sync(0xffffffff, tmp_b.x, i_shfl2_2); \
	tmp_b.y = __shfl_sync(0xffffffff, tmp_b.y, i_shfl2_2); \
	tmp_b.z = __shfl_sync(0xffffffff, tmp_b.z, i_shfl2_3); \
	tmp_b.w = __shfl_sync(0xffffffff, tmp_b.w, i_shfl2_3); \
}

#define G4(data)           \
{                           \
	COMPUTE \
    data[i4_0_0] = tmp_a.x; \
    data[i4_0_1] = tmp_a.y; \
    data[i4_1_0] = tmp_a.z; \
    data[i4_1_1] = tmp_a.w; \
    data[i4_2_0] = tmp_b.x; \
    data[i4_2_1] = tmp_b.y; \
    data[i4_3_0] = tmp_b.z; \
    data[i4_3_1] = tmp_b.w; \
    __syncwarp(); \
    tmp_a.x = data[i1_0_0]; \
    tmp_a.y = data[i1_0_1]; \
    tmp_a.z = data[i1_1_0]; \
    tmp_a.w = data[i1_1_1]; \
    tmp_b.x = data[i1_2_0]; \
    tmp_b.y = data[i1_2_1]; \
    tmp_b.z = data[i1_3_0]; \
    tmp_b.w = data[i1_3_1]; \
}

__constant__ int offsets[768] = {
		0, 4, 8, 12,
		1, 5, 9, 13,
		2, 6, 10, 14,
		3, 7, 11, 15,
		16, 20, 24, 28,
		17, 21, 25, 29,
		18, 22, 26, 30,
		19, 23, 27, 31,
		32, 36, 40, 44,
		33, 37, 41, 45,
		34, 38, 42, 46,
		35, 39, 43, 47,
		48, 52, 56, 60,
		49, 53, 57, 61,
		50, 54, 58, 62,
		51, 55, 59, 63,
		64, 68, 72, 76,
		65, 69, 73, 77,
		66, 70, 74, 78,
		67, 71, 75, 79,
		80, 84, 88, 92,
		81, 85, 89, 93,
		82, 86, 90, 94,
		83, 87, 91, 95,
		96, 100, 104, 108,
		97, 101, 105, 109,
		98, 102, 106, 110,
		99, 103, 107, 111,
		112, 116, 120, 124,
		113, 117, 121, 125,
		114, 118, 122, 126,
		115, 119, 123, 127,
		0, 5, 10, 15,
		1, 6, 11, 12,
		2, 7, 8, 13,
		3, 4, 9, 14,
		16, 21, 26, 31,
		17, 22, 27, 28,
		18, 23, 24, 29,
		19, 20, 25, 30,
		32, 37, 42, 47,
		33, 38, 43, 44,
		34, 39, 40, 45,
		35, 36, 41, 46,
		48, 53, 58, 63,
		49, 54, 59, 60,
		50, 55, 56, 61,
		51, 52, 57, 62,
		64, 69, 74, 79,
		65, 70, 75, 76,
		66, 71, 72, 77,
		67, 68, 73, 78,
		80, 85, 90, 95,
		81, 86, 91, 92,
		82, 87, 88, 93,
		83, 84, 89, 94,
		96, 101, 106, 111,
		97, 102, 107, 108,
		98, 103, 104, 109,
		99, 100, 105, 110,
		112, 117, 122, 127,
		113, 118, 123, 124,
		114, 119, 120, 125,
		115, 116, 121, 126,
		0, 32, 64, 96,
		1, 33, 65, 97,
		2, 34, 66, 98,
		3, 35, 67, 99,
		4, 36, 68, 100,
		5, 37, 69, 101,
		6, 38, 70, 102,
		7, 39, 71, 103,
		8, 40, 72, 104,
		9, 41, 73, 105,
		10, 42, 74, 106,
		11, 43, 75, 107,
		12, 44, 76, 108,
		13, 45, 77, 109,
		14, 46, 78, 110,
		15, 47, 79, 111,
		16, 48, 80, 112,
		17, 49, 81, 113,
		18, 50, 82, 114,
		19, 51, 83, 115,
		20, 52, 84, 116,
		21, 53, 85, 117,
		22, 54, 86, 118,
		23, 55, 87, 119,
		24, 56, 88, 120,
		25, 57, 89, 121,
		26, 58, 90, 122,
		27, 59, 91, 123,
		28, 60, 92, 124,
		29, 61, 93, 125,
		30, 62, 94, 126,
		31, 63, 95, 127,
		0, 33, 80, 113,
		1, 48, 81, 96,
		2, 35, 82, 115,
		3, 50, 83, 98,
		4, 37, 84, 117,
		5, 52, 85, 100,
		6, 39, 86, 119,
		7, 54, 87, 102,
		8, 41, 88, 121,
		9, 56, 89, 104,
		10, 43, 90, 123,
		11, 58, 91, 106,
		12, 45, 92, 125,
		13, 60, 93, 108,
		14, 47, 94, 127,
		15, 62, 95, 110,
		16, 49, 64, 97,
		17, 32, 65, 112,
		18, 51, 66, 99,
		19, 34, 67, 114,
		20, 53, 68, 101,
		21, 36, 69, 116,
		22, 55, 70, 103,
		23, 38, 71, 118,
		24, 57, 72, 105,
		25, 40, 73, 120,
		26, 59, 74, 107,
		27, 42, 75, 122,
		28, 61, 76, 109,
		29, 44, 77, 124,
		30, 63, 78, 111,
		31, 46, 79, 126,
        0, 1, 2, 3,
        1, 2, 3, 0,
        2, 3, 0, 1,
        3, 0, 1, 2,
        4, 5, 6, 7,
        5, 6, 7, 4,
        6, 7, 4, 5,
        7, 4, 5, 6,
        8, 9, 10, 11,
        9, 10, 11, 8,
        10, 11, 8, 9,
        11, 8, 9, 10,
        12, 13, 14, 15,
        13, 14, 15, 12,
        14, 15, 12, 13,
        15, 12, 13, 14,
        16, 17, 18, 19,
        17, 18, 19, 16,
        18, 19, 16, 17,
        19, 16, 17, 18,
        20, 21, 22, 23,
        21, 22, 23, 20,
        22, 23, 20, 21,
        23, 20, 21, 22,
        24, 25, 26, 27,
        25, 26, 27, 24,
        26, 27, 24, 25,
        27, 24, 25, 26,
        28, 29, 30, 31,
        29, 30, 31, 28,
        30, 31, 28, 29,
        31, 28, 29, 30,
        0, 1, 16, 17,
        1, 16, 17, 0,
        2, 3, 18, 19,
        3, 18, 19, 2,
        4, 5, 20, 21,
        5, 20, 21, 4,
        6, 7, 22, 23,
        7, 22, 23, 6,
        8, 9, 24, 25,
        9, 24, 25, 8,
        10, 11, 26, 27,
        11, 26, 27, 10,
        12, 13, 28, 29,
        13, 28, 29, 12,
        14, 15, 30, 31,
        15, 30, 31, 14,
        16, 17, 0, 1,
        17, 0, 1, 16,
        18, 19, 2, 3,
        19, 2, 3, 18,
        20, 21, 4, 5,
        21, 4, 5, 20,
        22, 23, 6, 7,
        23, 6, 7, 22,
        24, 25, 8, 9,
        25, 8, 9, 24,
        26, 27, 10, 11,
        27, 10, 11, 26,
        28, 29, 12, 13,
        29, 12, 13, 28,
        30, 31, 14, 15,
        31, 14, 15, 30
};

inline __host__ __device__ void operator^=( uint4& a, uint4 s) {
   a.x ^= s.x; a.y ^= s.y; a.z ^= s.z; a.w ^= s.w;
}

__global__ void fill_blocks(uint32_t *scratchpad0,
							uint32_t *scratchpad1,
							uint32_t *scratchpad2,
							uint32_t *scratchpad3,
							uint32_t *scratchpad4,
							uint32_t *scratchpad5,
							uint32_t *seed,
							uint32_t *out,
                            uint32_t *refs, // 32 bit
                            uint32_t *idxs, // first bit is keep flag, next 31 bit is current idx
							uint32_t *segments,
							int memsize,
							int lanes,
                            int seg_length,
                            int seg_count,
							int threads_per_chunk,
							int thread_idx) {
    extern __shared__ uint32_t shared[]; // lanes * BLOCK_SIZE_UINT [local state] + lanes * 32 [refs buffer] ( + lanes * 32 [idx buffer])

	uint32_t *local_state = shared;
	uint32_t *local_refs = shared + (lanes * BLOCK_SIZE_UINT);
	uint32_t *local_idxs = shared + (lanes * BLOCK_SIZE_UINT + lanes * 32);

	uint4 tmp_a, tmp_b, tmp_c, tmp_d, tmp_p, tmp_q, tmp_l, tmp_m;

	int hash = blockIdx.x;
	int mem_hash = hash + thread_idx;
	int local_id = threadIdx.x;
	int lane_length = seg_length * 4;

	int id = local_id % THREADS_PER_LANE;
	int lane = local_id / THREADS_PER_LANE;

	int offset = id << 2;

	int i1_0_0 = 2 * offsets[offset];
	int i1_0_1 = i1_0_0 + 1;
	int i1_1_0 = 2 * offsets[offset + 1];
	int i1_1_1 = i1_1_0 + 1;
	int i1_2_0 = 2 * offsets[offset + 2];
	int i1_2_1 = i1_2_0 + 1;
	int i1_3_0 = 2 * offsets[offset + 3];
	int i1_3_1 = i1_3_0 + 1;

	int i2_0_0 = 2 * offsets[offset + 128];
	int i2_0_1 = i2_0_0 + 1;
	int i2_1_0 = 2 * offsets[offset + 129];
	int i2_1_1 = i2_1_0 + 1;
	int i2_2_0 = 2 * offsets[offset + 130];
	int i2_2_1 = i2_2_0 + 1;
	int i2_3_0 = 2 * offsets[offset + 131];
	int i2_3_1 = i2_3_0 + 1;

	int i3_0_0 = 2 * offsets[offset + 256];
	int i3_0_1 = i3_0_0 + 1;
	int i3_1_0 = 2 * offsets[offset + 257];
	int i3_1_1 = i3_1_0 + 1;
	int i3_2_0 = 2 * offsets[offset + 258];
	int i3_2_1 = i3_2_0 + 1;
	int i3_3_0 = 2 * offsets[offset + 259];
	int i3_3_1 = i3_3_0 + 1;

	int i4_0_0 = 2 * offsets[offset + 384];
	int i4_0_1 = i4_0_0 + 1;
	int i4_1_0 = 2 * offsets[offset + 385];
	int i4_1_1 = i4_1_0 + 1;
	int i4_2_0 = 2 * offsets[offset + 386];
	int i4_2_1 = i4_2_0 + 1;
	int i4_3_0 = 2 * offsets[offset + 387];
	int i4_3_1 = i4_3_0 + 1;

	int i_shfl1_1 = offsets[offset + 513];
	int i_shfl1_2 = offsets[offset + 514];
	int i_shfl1_3 = offsets[offset + 515];
	int i_shfl2_1 = offsets[offset + 641];
	int i_shfl2_2 = offsets[offset + 642];
	int i_shfl2_3 = offsets[offset + 643];

    int scratchpad_location = mem_hash / threads_per_chunk;
    uint4 *memory = reinterpret_cast<uint4*>(scratchpad0);
    if(scratchpad_location == 1) memory = reinterpret_cast<uint4*>(scratchpad1);
    if(scratchpad_location == 2) memory = reinterpret_cast<uint4*>(scratchpad2);
    if(scratchpad_location == 3) memory = reinterpret_cast<uint4*>(scratchpad3);
    if(scratchpad_location == 4) memory = reinterpret_cast<uint4*>(scratchpad4);
    if(scratchpad_location == 5) memory = reinterpret_cast<uint4*>(scratchpad5);
    int hash_offset = mem_hash - scratchpad_location * threads_per_chunk;
    memory = memory + hash_offset * (memsize >> 4); // memsize / 16 -> 16 bytes in uint4

	uint32_t *mem_seed = seed + hash * lanes * 2 * BLOCK_SIZE_UINT;

	uint32_t *seed_src = mem_seed + lane * 2 * BLOCK_SIZE_UINT;
	uint4 *seed_dst = memory + lane * lane_length * BLOCK_SIZE_UINT4;

	seed_dst[id] = make_uint4(seed_src[i1_0_0], seed_src[i1_0_1], seed_src[i1_1_0], seed_src[i1_1_1]);
	seed_dst[id + 32] = make_uint4(seed_src[i1_2_0], seed_src[i1_2_1], seed_src[i1_3_0], seed_src[i1_3_1]);
	seed_src += BLOCK_SIZE_UINT;
	seed_dst += BLOCK_SIZE_UINT4;
	seed_dst[id] = make_uint4(seed_src[i1_0_0], seed_src[i1_0_1], seed_src[i1_1_0], seed_src[i1_1_1]);
	seed_dst[id + 32] = make_uint4(seed_src[i1_2_0], seed_src[i1_2_1], seed_src[i1_3_0], seed_src[i1_3_1]);

	uint4 *next_block;
	uint4 *prev_block;
	uint4 *ref_block;
    uint32_t *seg_refs, *seg_idxs;

	local_state = local_state + lane * BLOCK_SIZE_UINT;
	local_refs = local_refs + lane * 32;
	local_idxs = local_idxs + lane * 32;

    segments += (lane * 3);

	for(int s = 0; s < (seg_count / lanes); s++) {
		int idx = ((s == 0) ? 2 : 0); // index for first slice in each lane is 2
		int with_xor = ((s >= 4) ? 1 : 0);
		int keep = 1;
		int slice = s % 4;
		int pass = s / 4;

		uint32_t *cur_seg = &segments[s * lanes * 3];

		uint32_t cur_idx = cur_seg[0];
        uint32_t prev_idx = cur_seg[1];
        uint32_t seg_type = cur_seg[2];
        uint32_t ref_idx = 0;

        prev_block = memory + prev_idx * BLOCK_SIZE_UINT4;

        tmp_a = prev_block[id];
        tmp_b = prev_block[id + 32];

        __syncthreads();

        if(seg_type == 0) {
            seg_refs = refs + ((s * lanes + lane) * seg_length - ((s > 0) ? lanes : lane) * 2);
            if(idxs != NULL) seg_idxs = idxs + ((s * lanes + lane) * seg_length - ((s > 0) ? lanes : lane) * 2);

            for (cur_idx--;idx < seg_length; seg_refs += 32, seg_idxs += 32) {
				uint64_t i_limit = seg_length - idx;
				if (i_limit > 32) i_limit = 32;

				local_refs[id] = seg_refs[id];
				ref_idx = local_refs[0];

				if(idxs != NULL) {
					local_idxs[id] = seg_idxs[id];
					cur_idx = local_idxs[0];
					keep = cur_idx & 0x80000000;
					cur_idx = cur_idx & 0x7FFFFFFF;
				} else
				    cur_idx++;

                ref_block = memory + ref_idx * BLOCK_SIZE_UINT4;
                tmp_p = ref_block[id];
                tmp_q = ref_block[id + 32];

                for (int i = 0; i < i_limit; i++, idx++) {
                    next_block = memory + cur_idx * BLOCK_SIZE_UINT4;
					if(with_xor == 1) {
						tmp_l = next_block[id];
						tmp_m = next_block[id + 32];
					}

					tmp_a ^= tmp_p;
                    tmp_b ^= tmp_q;

                    if (i < (i_limit - 1)) {
						ref_idx = local_refs[i + 1];

						if(idxs != NULL) {
							cur_idx = local_idxs[i + 1];
							keep = cur_idx & 0x80000000;
							cur_idx = cur_idx & 0x7FFFFFFF;
						}
						else
							cur_idx++;

                        ref_block = memory + ref_idx * BLOCK_SIZE_UINT4;
                        tmp_p = ref_block[id];
                        tmp_q = ref_block[id + 32];
                    }

					tmp_c = tmp_a;
					tmp_d = tmp_b;

					G1(local_state);
                    G2(local_state);
                    G3(local_state);
                    G4(local_state);

                    if(with_xor == 1) {
						tmp_c ^= tmp_l;
						tmp_d ^= tmp_m;
					}

                    tmp_a ^= tmp_c;
                    tmp_b ^= tmp_d;

                    if(keep > 0) {
						next_block[id] = tmp_a;
						next_block[id + 32] = tmp_b;
					}
                }
            }
        }
        else {

            for (; idx < seg_length; idx++, cur_idx++) {
				next_block = memory + cur_idx * BLOCK_SIZE_UINT4;

				if(with_xor == 1) {
					tmp_l = next_block[id];
					tmp_m = next_block[id + 32];
				}

				uint32_t pseudo_rand_lo = __shfl_sync(0xffffffff, tmp_a.x, 0);

				if(lanes > 1) {
                    uint32_t pseudo_rand_hi = __shfl_sync(0xffffffff, tmp_a.y, 0);

                    uint64_t ref_lane = pseudo_rand_hi % lanes; // thr_cost
                    uint32_t reference_area_size = 0;
                    if (pass > 0) {
                        if (lane == ref_lane) {
                            reference_area_size = lane_length - seg_length + idx - 1;
                        } else {
                            reference_area_size = lane_length - seg_length + ((idx == 0) ? (-1) : 0);
                        }
                    } else {
                        if (lane == ref_lane) {
                            reference_area_size = slice * seg_length + idx - 1; // seg_length
                        } else {
                            reference_area_size = slice * seg_length + ((idx == 0) ? (-1) : 0);
                        }
                    }
                    asm("{mul.hi.u32 %0, %1, %1; mul.hi.u32 %0, %0, %2; }": "=r"(pseudo_rand_lo) : "r"(pseudo_rand_lo), "r"(reference_area_size));

                    uint32_t relative_position = reference_area_size - 1 - pseudo_rand_lo;

                    ref_idx = ref_lane * lane_length +
                              (((pass > 0 && slice < 3) ? ((slice + 1) * seg_length) : 0) + relative_position) %
                              lane_length;
                }
				else {
                    uint32_t reference_area_size = 0;
                    if (pass > 0) {
                        reference_area_size = lane_length - seg_length + idx - 1;
                    } else {
                        reference_area_size = slice * seg_length + idx - 1; // seg_length
                    }
                    asm("{mul.hi.u32 %0, %1, %1; mul.hi.u32 %0, %0, %2; }": "=r"(pseudo_rand_lo) : "r"(pseudo_rand_lo), "r"(reference_area_size));

                    uint32_t relative_position = reference_area_size - 1 - pseudo_rand_lo;

                    ref_idx = (((pass > 0 && slice < 3) ? ((slice + 1) * seg_length) : 0) + relative_position) %
                              lane_length;
				}

				ref_block = memory + ref_idx * BLOCK_SIZE_UINT4;

				tmp_a ^= ref_block[id];
				tmp_b ^= ref_block[id + 32];

				tmp_c = tmp_a;
				tmp_d = tmp_b;

				G1(local_state);
				G2(local_state);
				G3(local_state);
				G4(local_state);

				if(with_xor == 1) {
					tmp_c ^= tmp_l;
					tmp_d ^= tmp_m;
				}

				tmp_a ^= tmp_c;
				tmp_b ^= tmp_d;

				next_block[id] = tmp_a;
				next_block[id + 32] = tmp_b;
            }
        }
	}

    local_state[i1_0_0] = tmp_a.x;
    local_state[i1_0_1] = tmp_a.y;
    local_state[i1_1_0] = tmp_a.z;
    local_state[i1_1_1] = tmp_a.w;
    local_state[i1_2_0] = tmp_b.x;
    local_state[i1_2_1] = tmp_b.y;
    local_state[i1_3_0] = tmp_b.z;
    local_state[i1_3_1] = tmp_b.w;

    __syncthreads();

	// at this point local_state will contain the final blocks

	if(lane == 0) { // first lane needs to acumulate results
		tmp_a = make_uint4(0, 0, 0, 0);
		tmp_b = make_uint4(0, 0, 0, 0);

		for(int l=0; l<lanes; l++) {
			uint4 *block = (uint4 *)(shared + l * BLOCK_SIZE_UINT);
			tmp_a ^= block[id];
			tmp_b ^= block[id + 32];
		}

		uint4 *out_mem = (uint4 *)(out + hash * BLOCK_SIZE_UINT);
		out_mem[id] = tmp_a;
		out_mem[id + 32] = tmp_b;
	}
};

__global__ void prehash (
        uint32_t *preseed,
        uint32_t *seed,
		int memsz,
		int lanes,
		int passes,
		int pwdlen,
		int saltlen,
        int threads) { // len is given in uint32 units
    extern __shared__ uint32_t shared[]; // size = max(lanes * 2, 8) * 88

	int seeds_batch_size = blockDim.x / 4; // number of seeds per block
	int hash_batch_size = seeds_batch_size / (lanes * 2); // number of hashes per block

	int id = threadIdx.x; // minimum 32 threads
	int thr_id = id % 4; // thread id in session
	int session = id / 4; // blake2b hashing session

    int hash = blockIdx.x * hash_batch_size;
    int hash_idx = session / (lanes * 2);
    hash += hash_idx;

    if(hash < threads) {
        int hash_session = session % (lanes * 2); // session in hash

        int lane = hash_session / 2;  // 2 lanes
        int idx = hash_session % 2; // idx in lane

        uint32_t *local_mem = &shared[session * BLAKE_SHARED_MEM_UINT];
        uint32_t *local_seed = seed + (hash * lanes * 2 + hash_session) * BLOCK_SIZE_UINT;

        uint64_t *h = (uint64_t *) &local_mem[20];
        uint32_t *buf = (uint32_t *) &h[10];
        uint32_t *value = &buf[32];
        uint32_t *local_preseed = &value[1];

        uint32_t *cursor_in = preseed;
        uint32_t *cursor_out = local_preseed;

        for(int i=0; i < (pwdlen >> 2); i++, cursor_in += 4, cursor_out += 4) {
            cursor_out[thr_id] = cursor_in[thr_id];
        }

        if(thr_id == 0) {
            for (int i = 0; i < (pwdlen % 4); i++) {
                cursor_out[i] = cursor_in[i];
            }

            uint32_t nonce = (preseed[9] >> 24) | (preseed[10] << 8);
            nonce += hash;
            local_preseed[9] = (preseed[9] & 0x00FFFFFF) | (nonce << 24);
            local_preseed[10] = (preseed[10] & 0xFF000000) | (nonce >> 8);
        }

        int buf_len = blake2b_init(h, ARGON2_PREHASH_DIGEST_LENGTH_UINT, thr_id);
        *value = lanes; //lanes
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = 32; //outlen
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = memsz; //m_cost
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = passes; //t_cost
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = ARGON2_VERSION; //version
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = ARGON2_TYPE_VALUE; //type
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        *value = pwdlen * 4; //pw_len
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        buf_len = blake2b_update(local_preseed, pwdlen, h, buf, buf_len, thr_id);
        *value = saltlen * 4; //salt_len
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
		buf_len = blake2b_update(local_preseed, saltlen, h, buf, buf_len, thr_id);
        *value = 0; //secret_len
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        buf_len = blake2b_update(NULL, 0, h, buf, buf_len, thr_id);
        *value = 0; //ad_len
        buf_len = blake2b_update(value, 1, h, buf, buf_len, thr_id);
        buf_len = blake2b_update(NULL, 0, h, buf, buf_len, thr_id);

        blake2b_final(local_mem, ARGON2_PREHASH_DIGEST_LENGTH_UINT, h, buf, buf_len, thr_id);

        if (thr_id == 0) {
            local_mem[ARGON2_PREHASH_DIGEST_LENGTH_UINT] = idx;
            local_mem[ARGON2_PREHASH_DIGEST_LENGTH_UINT + 1] = lane;
        }

        blake2b_digestLong(local_seed, ARGON2_DWORDS_IN_BLOCK, local_mem, ARGON2_PREHASH_SEED_LENGTH_UINT, thr_id,
                           &local_mem[20]);
    }
}

__global__ void posthash (
        uint32_t *hash,
        uint32_t *out,
        uint32_t *preseed) {
    extern __shared__ uint32_t shared[]; // size = 120

    int hash_id = blockIdx.x;
    int thread = threadIdx.x;

    uint32_t *local_hash = hash + hash_id * ((ARGON2_RAW_LENGTH / 4) + 1);
    uint32_t *local_out = out + hash_id * BLOCK_SIZE_UINT;

    blake2b_digestLong(local_hash, ARGON2_RAW_LENGTH / 4, local_out, ARGON2_DWORDS_IN_BLOCK, thread, shared);

    if(thread == 0) {
        uint32_t nonce = (preseed[9] >> 24) | (preseed[10] << 8);
        nonce += hash_id;
        local_hash[ARGON2_RAW_LENGTH / 4] = nonce;
    }
}

void cuda_allocate(CudaDeviceInfo *device, double chunks, size_t chunk_size) {
	Argon2Profile *profile = device->profileInfo.profile;

	device->error = cudaSetDevice(device->cudaIndex);
	if(device->error != cudaSuccess) {
		device->errorMessage = "Error setting current device for memory allocation.";
		return;
	}

	size_t allocated_mem_for_current_chunk = 0;

	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_0, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_1, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_2, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_3, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_4, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	if (chunks > 0) {
		allocated_mem_for_current_chunk = chunks > 1 ? chunk_size : (size_t)ceil(chunk_size * chunks);
		chunks -= 1;
	}
	else {
		allocated_mem_for_current_chunk = 1;
	}
	device->error = cudaMalloc(&device->arguments.memoryChunk_5, allocated_mem_for_current_chunk);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}

	uint32_t *refs = (uint32_t *)malloc(profile->blockRefsSize * sizeof(uint32_t));
	for(int i=0;i<profile->blockRefsSize;i++) {
		refs[i] = profile->blockRefs[i*3 + 1];
	}

	device->error = cudaMalloc(&device->arguments.refs, profile->blockRefsSize * sizeof(uint32_t));
	if(device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}

	device->error = cudaMemcpy(device->arguments.refs, refs, profile->blockRefsSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->errorMessage = "Error copying memory.";
		return;
	}
	free(refs);

	if(profile->succesiveIdxs == 1) {
		device->arguments.idxs = NULL;
	}
	else {
		uint32_t *idxs = (uint32_t *) malloc(profile->blockRefsSize * sizeof(uint32_t));
		for (int i = 0; i < profile->blockRefsSize; i++) {
			idxs[i] = profile->blockRefs[i * 3];
			if (profile->blockRefs[i * 3 + 2] == 1) {
				idxs[i] |= 0x80000000;
			}
		}

		device->error = cudaMalloc(&device->arguments.idxs, profile->blockRefsSize * sizeof(uint32_t));
		if (device->error != cudaSuccess) {
			device->errorMessage = "Error allocating memory.";
			return;
		}

		device->error = cudaMemcpy(device->arguments.idxs, idxs, profile->blockRefsSize * sizeof(uint32_t),
								   cudaMemcpyHostToDevice);
		if (device->error != cudaSuccess) {
			device->errorMessage = "Error copying memory.";
			return;
		}
		free(idxs);
	}

	//reorganize segments data
	device->error = cudaMalloc(&device->arguments.segments, profile->segCount * 3 * sizeof(uint32_t));
	if(device->error != cudaSuccess) {
		device->errorMessage = "Error allocating memory.";
		return;
	}
	device->error = cudaMemcpy(device->arguments.segments, profile->segments, profile->segCount * 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	if(device->error != cudaSuccess) {
		device->errorMessage = "Error copying memory.";
		return;
	}

#ifdef PARALLEL_CUDA
	int threads = device->profileInfo.threads / 2;
#else
	int threads = device->profileInfo.threads;
#endif

	size_t preseed_memory_size = profile->pwdLen * 4;
	size_t seed_memory_size = threads * (profile->thrCost * 2) * ARGON2_BLOCK_SIZE;
	size_t out_memory_size = threads * ARGON2_BLOCK_SIZE;
	size_t hash_memory_size = threads * (xmrig::ARGON2_HASHLEN + 4);

    device->error = cudaMalloc(&device->arguments.preseedMemory[0], preseed_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.seedMemory[0], seed_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.outMemory[0], out_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.hashMemory[0], hash_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMallocHost(&device->arguments.hostSeedMemory[0], 132 * threads);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating pinned memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.preseedMemory[1], preseed_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.seedMemory[1], seed_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.outMemory[1], out_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMalloc(&device->arguments.hashMemory[1], hash_memory_size);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating memory.";
        return;
    }
    device->error = cudaMallocHost(&device->arguments.hostSeedMemory[1], 132 * threads);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error allocating pinned memory.";
        return;
    }
}

void cuda_free(CudaDeviceInfo *device) {
	cudaSetDevice(device->cudaIndex);

	if(device->arguments.idxs != NULL) {
		cudaFree(device->arguments.idxs);
		device->arguments.idxs = NULL;
	}

	if(device->arguments.refs != NULL) {
		cudaFree(device->arguments.refs);
		device->arguments.refs = NULL;
	}

	if(device->arguments.segments != NULL) {
		cudaFree(device->arguments.segments);
		device->arguments.segments = NULL;
	}

    if(device->arguments.memoryChunk_0 != NULL) {
        cudaFree(device->arguments.memoryChunk_0);
        device->arguments.memoryChunk_0 = NULL;
    }

    if(device->arguments.memoryChunk_1 != NULL) {
        cudaFree(device->arguments.memoryChunk_1);
        device->arguments.memoryChunk_1 = NULL;
    }

    if(device->arguments.memoryChunk_2 != NULL) {
        cudaFree(device->arguments.memoryChunk_2);
        device->arguments.memoryChunk_2 = NULL;
    }

    if(device->arguments.memoryChunk_3 != NULL) {
        cudaFree(device->arguments.memoryChunk_3);
        device->arguments.memoryChunk_3 = NULL;
    }

    if(device->arguments.memoryChunk_4 != NULL) {
        cudaFree(device->arguments.memoryChunk_4);
        device->arguments.memoryChunk_4 = NULL;
    }

    if(device->arguments.memoryChunk_5 != NULL) {
        cudaFree(device->arguments.memoryChunk_5);
        device->arguments.memoryChunk_5 = NULL;
    }

    if(device->arguments.preseedMemory != NULL) {
        for(int i=0;i<2;i++) {
            if(device->arguments.preseedMemory[i] != NULL)
                cudaFree(device->arguments.preseedMemory[i]);
            device->arguments.preseedMemory[i] = NULL;
        }
    }

	if(device->arguments.seedMemory != NULL) {
		for(int i=0;i<2;i++) {
			if(device->arguments.seedMemory[i] != NULL)
				cudaFree(device->arguments.seedMemory[i]);
			device->arguments.seedMemory[i] = NULL;
		}
	}

	if(device->arguments.outMemory != NULL) {
		for(int i=0;i<2;i++) {
			if(device->arguments.outMemory[i] != NULL)
				cudaFree(device->arguments.outMemory[i]);
			device->arguments.outMemory[i] = NULL;
		}
	}

    if(device->arguments.hashMemory != NULL) {
        for(int i=0;i<2;i++) {
            if(device->arguments.hashMemory[i] != NULL)
                cudaFree(device->arguments.hashMemory[i]);
            device->arguments.hashMemory[i] = NULL;
        }
    }

	if(device->arguments.hostSeedMemory != NULL) {
		for(int i=0;i<2;i++) {
			if(device->arguments.hostSeedMemory[i] != NULL)
				cudaFreeHost(device->arguments.hostSeedMemory[i]);
			device->arguments.hostSeedMemory[i] = NULL;
		}
	}

	cudaDeviceReset();
}

inline bool cudaCheckError(cudaError &err, string &errStr)
{
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        errStr = string("CUDA error: ") + cudaGetErrorString( err );
        return false;
    }

    return true;
}

bool cuda_kernel_prehasher(void *memory, int threads, Argon2Profile *profile, void *user_data) {
    CudaGpuMgmtThreadData *gpumgmt_thread = (CudaGpuMgmtThreadData *)user_data;
    CudaDeviceInfo *device = gpumgmt_thread->device;
    cudaStream_t stream = (cudaStream_t)gpumgmt_thread->deviceData;

    int sessions = max(profile->thrCost * 2, (uint32_t)8);
    double hashes_per_block = sessions / (profile->thrCost * 2.0);
    size_t work_items = sessions * 4;

    gpumgmt_thread->lock();

    memcpy(device->arguments.hostSeedMemory[gpumgmt_thread->threadId], memory, gpumgmt_thread->hashData.inSize);

    device->error = cudaMemcpyAsync(device->arguments.preseedMemory[gpumgmt_thread->threadId], device->arguments.hostSeedMemory[gpumgmt_thread->threadId], gpumgmt_thread->hashData.inSize, cudaMemcpyHostToDevice, stream);
    if (device->error != cudaSuccess) {
        device->errorMessage = "Error writing to gpu memory.";
        gpumgmt_thread->unlock();
        return false;
    }

	prehash <<< ceil(threads / hashes_per_block), work_items, sessions * BLAKE_SHARED_MEM, stream>>> (
			device->arguments.preseedMemory[gpumgmt_thread->threadId],
			device->arguments.seedMemory[gpumgmt_thread->threadId],
			profile->memCost,
			profile->thrCost,
			profile->segCount / (4 * profile->thrCost),
            gpumgmt_thread->hashData.inSize / 4,
			profile->saltLen,
            threads);

    bool success = cudaCheckError(device->error, device->errorMessage);
    if(!success) {
        gpumgmt_thread->unlock();
        return false;
    }

    return true;
}

void *cuda_kernel_filler(int threads, Argon2Profile *profile, void *user_data) {
	CudaGpuMgmtThreadData *gpumgmt_thread = (CudaGpuMgmtThreadData *)user_data;
	CudaDeviceInfo *device = gpumgmt_thread->device;
	cudaStream_t stream = (cudaStream_t)gpumgmt_thread->deviceData;

    size_t work_items = KERNEL_WORKGROUP_SIZE * profile->thrCost;
    size_t shared_mem = profile->thrCost * (ARGON2_BLOCK_SIZE + 128 + (profile->succesiveIdxs == 1 ? 128 : 0));

	fill_blocks <<<threads, work_items, shared_mem, stream>>> ((uint32_t*)device->arguments.memoryChunk_0,
			(uint32_t*)device->arguments.memoryChunk_1,
			(uint32_t*)device->arguments.memoryChunk_2,
			(uint32_t*)device->arguments.memoryChunk_3,
			(uint32_t*)device->arguments.memoryChunk_4,
			(uint32_t*)device->arguments.memoryChunk_5,
			device->arguments.seedMemory[gpumgmt_thread->threadId],
			device->arguments.outMemory[gpumgmt_thread->threadId],
			device->arguments.refs,
			device->arguments.idxs,
			device->arguments.segments,
			profile->memSize,
			profile->thrCost,
			profile->segSize,
			profile->segCount,
			device->profileInfo.threads_per_chunk,
            gpumgmt_thread->threadsIdx);

	bool success = cudaCheckError(device->error, device->errorMessage);
	if(!success) {
        gpumgmt_thread->unlock();
        return NULL;
    }

 	return  (void *)1;
}

bool cuda_kernel_posthasher(void *memory, int threads, Argon2Profile *profile, void *user_data) {
	CudaGpuMgmtThreadData *gpumgmt_thread = (CudaGpuMgmtThreadData *)user_data;
	CudaDeviceInfo *device = gpumgmt_thread->device;
	cudaStream_t stream = (cudaStream_t)gpumgmt_thread->deviceData;

    size_t work_items = 4;

	posthash <<<threads, work_items, BLAKE_SHARED_MEM, stream>>> (
            device->arguments.hashMemory[gpumgmt_thread->threadId],
            device->arguments.outMemory[gpumgmt_thread->threadId],
            device->arguments.preseedMemory[gpumgmt_thread->threadId]);

    if(!cudaCheckError(device->error, device->errorMessage)) {
        gpumgmt_thread->unlock();
        return false;
    }

	device->error = cudaMemcpyAsync(device->arguments.hostSeedMemory[gpumgmt_thread->threadId], device->arguments.hashMemory[gpumgmt_thread->threadId], threads * (xmrig::ARGON2_HASHLEN + 4), cudaMemcpyDeviceToHost, stream);
	if (device->error != cudaSuccess) {
		device->errorMessage = "Error reading gpu memory.";
		gpumgmt_thread->unlock();
		return false;
	}

    cudaStreamSynchronize(stream);

    bool success = cudaCheckError(device->error, device->errorMessage);
    if(!success) {
        gpumgmt_thread->unlock();
        return false;
    }

    memcpy(memory, device->arguments.hostSeedMemory[gpumgmt_thread->threadId], threads * (xmrig::ARGON2_HASHLEN + 4));
	gpumgmt_thread->unlock();

	return true;
}