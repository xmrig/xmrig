#define BLOCK_BYTES	32
#define OUT_BYTES	16
#define BLAKE_SHARED_MEM            480
#define BLAKE_SHARED_MEM_UINT       120

#define G(m, r, i, a, b, c, d) \
do { \
	a = a + b + m[blake2b_sigma[r][2 * i + 0]]; \
	d = rotr64(d ^ a, 32); \
	c = c + d; \
	b = rotr64(b ^ c, 24); \
	a = a + b + m[blake2b_sigma[r][2 * i + 1]]; \
	d = rotr64(d ^ a, 16); \
	c = c + d; \
	b = rotr64(b ^ c, 63); \
} while ((void)0, 0)

#define G_S(m, a, b, c, d) \
do { \
	a = a + b + m; \
	d = rotr64(d ^ a, 32); \
	c = c + d; \
	b = rotr64(b ^ c, 24); \
	a = a + b + m; \
	d = rotr64(d ^ a, 16); \
	c = c + d; \
	b = rotr64(b ^ c, 63); \
} while ((void)0, 0)

#define ROUND(m, t, r) \
do { \
	G(m, r, t, v0, v1, v2, v3); \
    v1 = __shfl_sync(0xFFFFFFFF, v1, t + 1, 4); \
    v2 = __shfl_sync(0xFFFFFFFF, v2, t + 2, 4); \
    v3 = __shfl_sync(0xFFFFFFFF, v3, t + 3, 4); \
	G(m, r, (t + 4), v0, v1, v2, v3); \
    v1 = __shfl_sync(0xFFFFFFFF, v1, t + 3, 4); \
    v2 = __shfl_sync(0xFFFFFFFF, v2, t + 2, 4); \
    v3 = __shfl_sync(0xFFFFFFFF, v3, t + 1, 4); \
} while ((void)0, 0)

#define ROUND_S(m, t) \
do { \
	G_S(m, v0, v1, v2, v3); \
    v1 = __shfl_sync(0xFFFFFFFF, v1, t + 1, 4); \
    v2 = __shfl_sync(0xFFFFFFFF, v2, t + 2, 4); \
    v3 = __shfl_sync(0xFFFFFFFF, v3, t + 3, 4); \
	G_S(m, v0, v1, v2, v3); \
    v1 = __shfl_sync(0xFFFFFFFF, v1, t + 3, 4); \
    v2 = __shfl_sync(0xFFFFFFFF, v2, t + 2, 4); \
    v3 = __shfl_sync(0xFFFFFFFF, v3, t + 1, 4); \
} while ((void)0, 0)

__constant__ uint64_t blake2b_IV[8] = {
    0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
    0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
};

__constant__ uint32_t blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

__device__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ __forceinline__ void blake2b_compress(uint64_t *h, uint64_t *m, uint64_t f0, int thr_id)
{
    uint64_t v0, v1, v2, v3;

    v0 = h[thr_id];
    v1 = h[thr_id + 4];
    v2 = blake2b_IV[thr_id];
    v3 = blake2b_IV[thr_id + 4];

    if(thr_id == 0) v3 ^= h[8];
    if(thr_id == 1) v3 ^= h[9];
    if(thr_id == 2) v3 ^= f0;

    ROUND(m, thr_id, 0);
    ROUND(m, thr_id, 1);
    ROUND(m, thr_id, 2);
    ROUND(m, thr_id, 3);
    ROUND(m, thr_id, 4);
    ROUND(m, thr_id, 5);
    ROUND(m, thr_id, 6);
    ROUND(m, thr_id, 7);
    ROUND(m, thr_id, 8);
    ROUND(m, thr_id, 9);
    ROUND(m, thr_id, 10);
    ROUND(m, thr_id, 11);

    h[thr_id] ^= v0 ^ v2;
    h[thr_id + 4] ^= v1 ^ v3;
}

__device__ __forceinline__ void blake2b_compress_static(uint64_t *h, uint64_t m, uint64_t f0, int thr_id)
{
    uint64_t v0, v1, v2, v3;

    v0 = h[thr_id];
    v1 = h[thr_id + 4];
    v2 = blake2b_IV[thr_id];
    v3 = blake2b_IV[thr_id + 4];

    if(thr_id == 0) v3 ^= h[8];
    if(thr_id == 1) v3 ^= h[9];
    if(thr_id == 2) v3 ^= f0;

    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);
    ROUND_S(m, thr_id);

    h[thr_id] ^= v0 ^ v2;
    h[thr_id + 4] ^= v1 ^ v3;
}

__device__ __forceinline__ int blake2b_init(uint64_t *h, int out_len, int thr_id)
{
    h[thr_id * 2] = blake2b_IV[thr_id * 2];
    h[thr_id * 2 + 1] = blake2b_IV[thr_id * 2 + 1];

    if(thr_id == 0) {
        h[8] = h[9] = 0;
        h[0] = 0x6A09E667F3BCC908 ^ ((out_len * 4) | (1 << 16) | (1 << 24));
    }

    return 0;
}

__device__ __forceinline__ void blake2b_incrementCounter(uint64_t *h, int inc)
{
    h[8] += (inc * 4);
    h[9] += (h[8] < (inc * 4));
}

__device__ __forceinline__ int blake2b_update(uint32_t *in, int in_len, uint64_t *h, uint32_t *buf, int buf_len, int thr_id)
{
    uint32_t *cursor_in = in;
    uint32_t *cursor_out = buf + buf_len;

    if (buf_len + in_len > BLOCK_BYTES) {
        int left = BLOCK_BYTES - buf_len;

        for(int i=0; i < (left >> 2); i++, cursor_in += 4, cursor_out += 4) {
            cursor_out[thr_id] = cursor_in[thr_id];
        }

        if(thr_id == 0) {
            for (int i = 0; i < (left % 4); i++) {
                cursor_out[i] = cursor_in[i];
            }
            blake2b_incrementCounter(h, BLOCK_BYTES);
        }

        blake2b_compress(h, (uint64_t*)buf, 0, thr_id);

        buf_len = 0;

        in_len -= left;
        in += left;

        while (in_len > BLOCK_BYTES) {
            if(thr_id == 0)
                blake2b_incrementCounter(h, BLOCK_BYTES);

            cursor_in = in;
            cursor_out = buf;

            for(int i=0; i < (BLOCK_BYTES / 4); i++, cursor_in += 4, cursor_out += 4) {
                cursor_out[thr_id] = cursor_in[thr_id];
            }

            blake2b_compress(h, (uint64_t *)buf, 0, thr_id);

            in_len -= BLOCK_BYTES;
            in += BLOCK_BYTES;
        }
    }

    cursor_in = in;
    cursor_out = buf + buf_len;

    for(int i=0; i < (in_len >> 2); i++, cursor_in += 4, cursor_out += 4) {
        cursor_out[thr_id] = cursor_in[thr_id];
    }

    if(thr_id == 0) {
        for (int i = 0; i < (in_len % 4); i++) {
            cursor_out[i] = cursor_in[i];
        }
    }

    return buf_len + in_len;
}

__device__ __forceinline__ int blake2b_update_static(uint32_t in, int in_len, uint64_t *h, uint32_t *buf, int buf_len, int thr_id)
{
    uint64_t in64 = in;
    in64 = in64 << 32;
    in64 = in64 | in;

    uint32_t *cursor_out = buf + buf_len;

    if (buf_len + in_len > BLOCK_BYTES) {
        int left = BLOCK_BYTES - buf_len;

        for(int i=0; i < (left >> 2); i++, cursor_out += 4) {
            cursor_out[thr_id] = in;
        }

        if(thr_id == 0) {
            for (int i = 0; i < (left % 4); i++) {
                cursor_out[i] = in;
            }
            blake2b_incrementCounter(h, BLOCK_BYTES);
        }

        blake2b_compress(h, (uint64_t*)buf, 0, thr_id);

        buf_len = 0;

        in_len -= left;

        while (in_len > BLOCK_BYTES) {
            if(thr_id == 0)
                blake2b_incrementCounter(h, BLOCK_BYTES);

            blake2b_compress_static(h, in64, 0, thr_id);

            in_len -= BLOCK_BYTES;
        }
    }

    cursor_out = buf + buf_len;

    for(int i=0; i < (in_len >> 2); i++, cursor_out += 4) {
        cursor_out[thr_id] = in;
    }

    if(thr_id == 0) {
        for (int i = 0; i < (in_len % 4); i++) {
            cursor_out[i] = in;
        }
    }

    return buf_len + in_len;
}

__device__ __forceinline__ void blake2b_final(uint32_t *out, int out_len, uint64_t *h, uint32_t *buf, int buf_len, int thr_id)
{
    int left = BLOCK_BYTES - buf_len;
    uint32_t *cursor_out = buf + buf_len;

    for(int i=0; i < (left >> 2); i++, cursor_out += 4) {
        cursor_out[thr_id] = 0;
    }

    if(thr_id == 0) {
        for (int i = 0; i < (left % 4); i++) {
            cursor_out[i] = 0;
        }
        blake2b_incrementCounter(h, buf_len);
    }

    blake2b_compress(h, (uint64_t*)buf, 0xFFFFFFFFFFFFFFFF, thr_id);

    __syncthreads();

    uint32_t *cursor_in = (uint32_t *)h;
    cursor_out = out;

    for(int i=0; i < (out_len >> 2); i++, cursor_in += 4, cursor_out += 4) {
        cursor_out[thr_id] = cursor_in[thr_id];
    }

    if(thr_id == 0) {
        for (int i = 0; i < (out_len % 4); i++) {
            cursor_out[i] = cursor_in[i];
        }
    }
}

__device__ void blake2b_digestLong(uint32_t *out, int out_len, uint32_t *in, int in_len, int thr_id, uint32_t *shared)
{
    uint64_t *h = (uint64_t*)shared;
    uint32_t *buf = (uint32_t*)&h[10];
    uint32_t *out_buffer = &buf[32];
    int buf_len;

    if(thr_id == 0) buf[0] = (out_len * 4);
    buf_len = 1;

    if (out_len <= OUT_BYTES) {
        blake2b_init(h, out_len, thr_id);
        buf_len = blake2b_update(in, in_len, h, buf, buf_len, thr_id);
        blake2b_final(out, out_len, h, buf, buf_len, thr_id);
    } else {
        uint32_t *cursor_in = out_buffer;
        uint32_t *cursor_out = out;

        blake2b_init(h, OUT_BYTES, thr_id);
        buf_len = blake2b_update(in, in_len, h, buf, buf_len, thr_id);
        blake2b_final(out_buffer, OUT_BYTES, h, buf, buf_len, thr_id);

        for(int i=0; i < (OUT_BYTES / 8); i++, cursor_in += 4, cursor_out += 4) {
            cursor_out[thr_id] = cursor_in[thr_id];
        }

        out += OUT_BYTES / 2;

        int to_produce = out_len - OUT_BYTES / 2;
        while (to_produce > OUT_BYTES) {
            buf_len = blake2b_init(h, OUT_BYTES, thr_id);
            buf_len = blake2b_update(out_buffer, OUT_BYTES, h, buf, buf_len, thr_id);
            blake2b_final(out_buffer, OUT_BYTES, h, buf, buf_len, thr_id);

            cursor_out = out;
            cursor_in = out_buffer;
            for(int i=0; i < (OUT_BYTES / 8); i++, cursor_in += 4, cursor_out += 4) {
                cursor_out[thr_id] = cursor_in[thr_id];
            }

            out += OUT_BYTES / 2;
            to_produce -= OUT_BYTES / 2;
        }

        buf_len = blake2b_init(h, to_produce, thr_id);
        buf_len = blake2b_update(out_buffer, OUT_BYTES, h, buf, buf_len, thr_id);
        blake2b_final(out, to_produce, h, buf, buf_len, thr_id);
    }
}