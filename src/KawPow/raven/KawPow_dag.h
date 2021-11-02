#define ETHASH_HASH_BYTES 64
#define ETHASH_DATASET_PARENTS 512

#if (__CUDACC_VER_MAJOR__ > 8)
#define SHFL(x, y, z) __shfl_sync(0xFFFFFFFF, (x), (y), (z))
#else
#define SHFL(x, y, z) __shfl((x), (y), (z))
#endif

typedef union {
	uint32_t words[64 / sizeof(uint32_t)];
	uint2	 uint2s[64 / sizeof(uint2)];
	uint4	 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union {
	uint32_t words[200 / sizeof(uint32_t)];
	uint64_t uint64s[200 / sizeof(uint64_t)];
	uint2	 uint2s[200 / sizeof(uint2)];
	uint4	 uint4s[200 / sizeof(uint4)];
} hash200_t;

// Implementation based on:
// https://github.com/mjosaarinen/tiny_sha3/blob/master/sha3.c
// converted from 64->32 bit words

__device__ __constant__ const uint64_t keccakf_rndc[24] = {
	0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
	0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
	0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
	0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
	0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
	0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
	0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
	0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __forceinline__ uint64_t ROTL64(const uint64_t x, const int offset)
{
    uint64_t result;
    asm("{\n\t"
        ".reg .b64 lhs;\n\t"
        ".reg .u32 roff;\n\t"
        "shl.b64 lhs, %1, %2;\n\t"
        "sub.u32 roff, 64, %2;\n\t"
        "shr.b64 %0, %1, roff;\n\t"
        "add.u64 %0, lhs, %0;\n\t"
        "}\n"
        : "=l"(result)
        : "l"(x), "r"(offset));
    return result;
}

__device__ __forceinline__ void keccak_f1600_round(uint64_t st[25], const int r)
{
	const uint32_t keccakf_rotc[24] = {
		1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
		27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
	};
	const uint32_t keccakf_piln[24] = {
		10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
		15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
	};

	uint64_t t, bc[5];
	// Theta
	for (int i = 0; i < 5; i++)
		bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

	for (int i = 0; i < 5; i++) {
		t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
		for (uint32_t j = 0; j < 25; j += 5)
			st[j + i] ^= t;
	}

	// Rho Pi
	t = st[1];
	for (int i = 0; i < 24; i++) {
		uint32_t j = keccakf_piln[i];
		bc[0] = st[j];
		st[j] = ROTL64(t, keccakf_rotc[i]);
		t = bc[0];
	}

	//  Chi
	for (uint32_t j = 0; j < 25; j += 5) {
		for (int i = 0; i < 5; i++)
			bc[i] = st[j + i];
		for (int i = 0; i < 5; i++)
			st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
	}

	//  Iota
	st[0] ^= keccakf_rndc[r];
}

__device__ __forceinline__ void keccak_f1600(uint64_t st[25])
{
	for (int i = 8; i < 25; i++) {
		st[i] = 0;
	}
	st[8] = 0x8000000000000001;

	for (int r = 0; r < 24; r++) {
		keccak_f1600_round(st, r);
	}
}

#define FNV_PRIME	0x01000193U
#define fnv(x,y) ((uint32_t(x) * (FNV_PRIME)) ^uint32_t(y))
__device__ uint4 fnv4(uint4 a, uint4 b)
{
	uint4 c;
	c.x = a.x * FNV_PRIME ^ b.x;
	c.y = a.y * FNV_PRIME ^ b.y;
	c.z = a.z * FNV_PRIME ^ b.z;
	c.w = a.w * FNV_PRIME ^ b.w;
	return c;
}

#define NODE_WORDS (ETHASH_HASH_BYTES/sizeof(uint32_t))

__device__ __forceinline__ uint32_t fast_mod(uint32_t a, uint4 d)
{
	const uint64_t t = a;
	const uint32_t q = ((t + d.y) * d.x) >> d.z;
	return a - q * d.w;
}

__global__ void ethash_calculate_dag_item(uint32_t start, hash64_t *g_dag, uint64_t dag_bytes, hash64_t* g_light, uint4 light_words)
{
	uint64_t const node_index = start + uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;

	uint64_t num_nodes = dag_bytes / sizeof(hash64_t);
	uint64_t num_nodes_rounded = ((num_nodes + 3) / 4) * 4;

	if (node_index >= num_nodes_rounded) return; // None of the threads from this quad have valid node_index

	hash200_t dag_node;
	for(int i = 0; i < 4; ++i) {
		dag_node.uint4s[i] = g_light[fast_mod(node_index, light_words)].uint4s[i];
	}

	dag_node.words[0] ^= node_index;
	keccak_f1600(dag_node.uint64s);

	const int thread_id = threadIdx.x & 3;

	#pragma unroll(4)
	for (uint32_t i = 0; i < ETHASH_DATASET_PARENTS; ++i) {
		uint32_t parent_index = fast_mod(fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]), light_words);

		#pragma unroll
		for (uint32_t t = 0; t < 4; ++t) {

			const uint32_t shuffle_index = SHFL(parent_index, t, 4);

			const uint4 p4 = g_light[shuffle_index].uint4s[thread_id];

			#pragma unroll
			for (int w = 0; w < 4; ++w) {

				const uint4 s4 = make_uint4(SHFL(p4.x, w, 4), SHFL(p4.y, w, 4), SHFL(p4.z, w, 4), SHFL(p4.w, w, 4));
				if (t == thread_id) {
					dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], s4);
				}
			}
		}
	}

	keccak_f1600(dag_node.uint64s);

	for (uint32_t t = 0; t < 4; ++t) {
		uint32_t shuffle_index = SHFL(node_index, t, 4);
		uint4 s[4];
		for (uint32_t w = 0; w < 4; w++) {
			s[w] = make_uint4(SHFL(dag_node.uint4s[w].x, t, 4),
							  SHFL(dag_node.uint4s[w].y, t, 4),
							  SHFL(dag_node.uint4s[w].z, t, 4),
							  SHFL(dag_node.uint4s[w].w, t, 4));
		}
		if (shuffle_index * sizeof(hash64_t) < dag_bytes) {
			g_dag[shuffle_index].uint4s[thread_id] = s[thread_id];
		}
	}
}
