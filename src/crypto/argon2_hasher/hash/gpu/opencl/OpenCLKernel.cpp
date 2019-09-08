//
// Created by Haifa Bogdan Adnan on 06/08/2018.
//

#include "../../../common/common.h"

#include "OpenCLKernel.h"

string OpenCLKernel = R"OCL(
#define THREADS_PER_LANE               32
#define BLOCK_SIZE_ULONG                128
#define BLOCK_SIZE_UINT                 256
#define ARGON2_PREHASH_DIGEST_LENGTH_UINT   16
#define ARGON2_PREHASH_SEED_LENGTH_UINT     18

#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_DWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 4)

#define BLAKE_SHARED_MEM_ULONG       76

#define ARGON2_RAW_LENGTH           8

#define ARGON2_TYPE_VALUE               2
#define ARGON2_VERSION                  0x13

#define BLOCK_BYTES	32
#define OUT_BYTES	16

#ifdef USE_AMD_BITALIGN
#pragma OPENCL EXTENSION cl_amd_media_ops : enable

#define rotr64(x, n)       ((n) < 32 ? (amd_bitalign((uint)((x) >> 32), (uint)(x), (uint)(n)) | ((ulong)amd_bitalign((uint)(x), (uint)((x) >> 32), (uint)(n)) << 32)) : rotate((x), 64UL - (n)))
#else
#define rotr64(x, n)        rotate((x), 64UL - (n))
#endif

#define G(m, r, i, a, b, c, d) \
{ \
	a = a + b + m[blake2b_sigma[r][2 * i + 0]]; \
	d = rotr64(d ^ a, 32); \
	c = c + d; \
	b = rotr64(b ^ c, 24); \
	a = a + b + m[blake2b_sigma[r][2 * i + 1]]; \
	d = rotr64(d ^ a, 16); \
	c = c + d; \
	b = rotr64(b ^ c, 63); \
}

#define G_S(m, a, b, c, d) \
{ \
	a = a + b + m; \
	d = rotr64(d ^ a, 32); \
	c = c + d; \
	b = rotr64(b ^ c, 24); \
	a = a + b + m; \
	d = rotr64(d ^ a, 16); \
	c = c + d; \
	b = rotr64(b ^ c, 63); \
}

#define ROUND(m, t, r, shfl) \
{ \
	G(m, r, t, v0, v1, v2, v3); \
    shfl[t + 4] = v1; \
    shfl[t + 8] = v2; \
    shfl[t + 12] = v3; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    v1 = shfl[((t + 1) % 4)+ 4]; \
    v2 = shfl[((t + 2) % 4)+ 8]; \
    v3 = shfl[((t + 3) % 4)+ 12]; \
	G(m, r, (t + 4), v0, v1, v2, v3); \
    shfl[((t + 1) % 4)+ 4] = v1; \
    shfl[((t + 2) % 4)+ 8] = v2; \
    shfl[((t + 3) % 4)+ 12] = v3; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    v1 = shfl[t + 4]; \
    v2 = shfl[t + 8]; \
    v3 = shfl[t + 12]; \
}

#define ROUND_S(m, t, shfl) \
{ \
	G_S(m, v0, v1, v2, v3); \
    shfl[t + 4] = v1; \
    shfl[t + 8] = v2; \
    shfl[t + 12] = v3; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    v1 = shfl[((t + 1) % 4)+ 4]; \
    v2 = shfl[((t + 2) % 4)+ 8]; \
    v3 = shfl[((t + 3) % 4)+ 12]; \
	G_S(m, v0, v1, v2, v3); \
    shfl[((t + 1) % 4)+ 4] = v1; \
    shfl[((t + 2) % 4)+ 8] = v2; \
    shfl[((t + 3) % 4)+ 12] = v3; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    v1 = shfl[t + 4]; \
    v2 = shfl[t + 8]; \
    v3 = shfl[t + 12]; \
}

__constant ulong blake2b_IV[8] = {
        0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
        0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
        0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
        0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
};

__constant uint blake2b_sigma[12][16] = {
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

void blake2b_compress(__local ulong *h, __local ulong *m, ulong f0, __local ulong *shfl, int thr_id)
{
    ulong v0, v1, v2, v3;

    barrier(CLK_LOCAL_MEM_FENCE);

    v0 = h[thr_id];
    v1 = h[thr_id + 4];
    v2 = blake2b_IV[thr_id];
    v3 = blake2b_IV[thr_id + 4];

    if(thr_id == 0) v3 ^= h[8];
    if(thr_id == 1) v3 ^= h[9];
    if(thr_id == 2) v3 ^= f0;

    ROUND(m, thr_id, 0, shfl);
    ROUND(m, thr_id, 1, shfl);
    ROUND(m, thr_id, 2, shfl);
    ROUND(m, thr_id, 3, shfl);
    ROUND(m, thr_id, 4, shfl);
    ROUND(m, thr_id, 5, shfl);
    ROUND(m, thr_id, 6, shfl);
    ROUND(m, thr_id, 7, shfl);
    ROUND(m, thr_id, 8, shfl);
    ROUND(m, thr_id, 9, shfl);
    ROUND(m, thr_id, 10, shfl);
    ROUND(m, thr_id, 11, shfl);

    h[thr_id] ^= v0 ^ v2;
    h[thr_id + 4] ^= v1 ^ v3;
}

void blake2b_compress_static(__local ulong *h, ulong m, ulong f0, __local ulong *shfl, int thr_id)
{
    ulong v0, v1, v2, v3;

    barrier(CLK_LOCAL_MEM_FENCE);

    v0 = h[thr_id];
    v1 = h[thr_id + 4];
    v2 = blake2b_IV[thr_id];
    v3 = blake2b_IV[thr_id + 4];

    if(thr_id == 0) v3 ^= h[8];
    if(thr_id == 1) v3 ^= h[9];
    if(thr_id == 2) v3 ^= f0;

    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);
    ROUND_S(m, thr_id, shfl);

    h[thr_id] ^= v0 ^ v2;
    h[thr_id + 4] ^= v1 ^ v3;
}

void blake2b_incrementCounter(__local ulong *h, int inc)
{
    h[8] += (inc * 4);
    h[9] += (h[8] < (inc * 4));
}

void blake2b_final_global(__global uint *out, int out_len, __local ulong *h, __local uint *buf, int buf_len, __local ulong *shfl, int thr_id)
{
    int left = BLOCK_BYTES - buf_len;
    __local uint *cursor_out_local = buf + buf_len;

    for(int i=0; i < (left >> 2); i++, cursor_out_local += 4) {
        cursor_out_local[thr_id] = 0;
    }

    if(thr_id == 0) {
        for (int i = 0; i < (left % 4); i++) {
            cursor_out_local[i] = 0;
        }
        blake2b_incrementCounter(h, buf_len);
    }

    blake2b_compress(h, (__local ulong *)buf, 0xFFFFFFFFFFFFFFFF, shfl, thr_id);

    __local uint *cursor_in = (__local uint *)h;
    __global uint *cursor_out_global = out;

    for(int i=0; i < (out_len >> 2); i++, cursor_in += 4, cursor_out_global += 4) {
        cursor_out_global[thr_id] = cursor_in[thr_id];
    }

    if(thr_id == 0) {
        for (int i = 0; i < (out_len % 4); i++) {
            cursor_out_global[i] = cursor_in[i];
        }
    }
}

void blake2b_final_local(__local uint *out, int out_len, __local ulong *h, __local uint *buf, int buf_len, __local ulong *shfl, int thr_id)
{
    int left = BLOCK_BYTES - buf_len;
    __local uint *cursor_out = buf + buf_len;

    for(int i=0; i < (left >> 2); i++, cursor_out += 4) {
        cursor_out[thr_id] = 0;
    }

    if(thr_id == 0) {
        for (int i = 0; i < (left % 4); i++) {
            cursor_out[i] = 0;
        }
        blake2b_incrementCounter(h, buf_len);
    }

    blake2b_compress(h, (__local ulong *)buf, 0xFFFFFFFFFFFFFFFF, shfl, thr_id);

    __local uint *cursor_in = (__local uint *)h;
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

int blake2b_update_global(__global uint *in, int in_len, __local ulong *h, __local uint *buf, int buf_len, __local ulong *shfl, int thr_id)
{
    __global uint *cursor_in = in;
    __local uint *cursor_out = buf + buf_len;

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

        blake2b_compress(h, (__local ulong *)buf, 0, shfl, thr_id);

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

            blake2b_compress(h, (__local ulong *)buf, 0, shfl, thr_id);

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

int blake2b_update_static(uint in, int in_len, __local ulong *h, __local uint *buf, int buf_len, __local ulong *shfl, int thr_id)
{
    ulong in64 = in;
    in64 = in64 << 32;
    in64 = in64 | in;

    __local uint *cursor_out = buf + buf_len;

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

        blake2b_compress(h, (__local ulong *)buf, 0, shfl, thr_id);

        buf_len = 0;

        in_len -= left;

        while (in_len > BLOCK_BYTES) {
            if(thr_id == 0)
                blake2b_incrementCounter(h, BLOCK_BYTES);

            blake2b_compress_static(h, in64, 0, shfl, thr_id);

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

int blake2b_update_local(__local uint *in, int in_len, __local ulong *h, __local uint *buf, int buf_len, __local ulong *shfl, int thr_id)
{
    __local uint *cursor_in = in;
    __local uint *cursor_out = buf + buf_len;

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

        blake2b_compress(h, (__local ulong *)buf, 0, shfl, thr_id);

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

            blake2b_compress(h, (__local ulong *)buf, 0, shfl, thr_id);

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

int blake2b_init(__local ulong *h, int out_len, int thr_id)
{
    h[thr_id * 2] = blake2b_IV[thr_id * 2];
    h[thr_id * 2 + 1] = blake2b_IV[thr_id * 2 + 1];

    if(thr_id == 0) {
        h[8] = h[9] = 0;
        h[0] = 0x6A09E667F3BCC908 ^ ((out_len * 4) | (1 << 16) | (1 << 24));
    }

    return 0;
}

void blake2b_digestLong_global(__global uint *out, int out_len,
                       __global uint *in, int in_len,
                       int thr_id, __local ulong* shared)
{
    __local ulong *h = shared;
	__local ulong *shfl = &h[10];
    __local uint *buf = (__local uint *)&shfl[16];
    __local uint *out_buffer = &buf[32];
    int buf_len;

    if(thr_id == 0) buf[0] = (out_len * 4);
    buf_len = 1;

    if (out_len <= OUT_BYTES) {
        blake2b_init(h, out_len, thr_id);
        buf_len = blake2b_update_global(in, in_len, h, buf, buf_len, shfl, thr_id);
        blake2b_final_global(out, out_len, h, buf, buf_len, shfl, thr_id);
    } else {
        __local uint *cursor_in = out_buffer;
        __global uint *cursor_out = out;

        blake2b_init(h, OUT_BYTES, thr_id);
        buf_len = blake2b_update_global(in, in_len, h, buf, buf_len, shfl, thr_id);
        blake2b_final_local(out_buffer, OUT_BYTES, h, buf, buf_len, shfl, thr_id);

        for(int i=0; i < (OUT_BYTES / 8); i++, cursor_in += 4, cursor_out += 4) {
            cursor_out[thr_id] = cursor_in[thr_id];
        }

        out += OUT_BYTES / 2;

        int to_produce = out_len - OUT_BYTES / 2;
        while (to_produce > OUT_BYTES) {
            buf_len = blake2b_init(h, OUT_BYTES, thr_id);
            buf_len = blake2b_update_local(out_buffer, OUT_BYTES, h, buf, buf_len, shfl, thr_id);
            blake2b_final_local(out_buffer, OUT_BYTES, h, buf, buf_len, shfl, thr_id);

            cursor_out = out;
            cursor_in = out_buffer;
            for(int i=0; i < (OUT_BYTES / 8); i++, cursor_in += 4, cursor_out += 4) {
                cursor_out[thr_id] = cursor_in[thr_id];
            }

            out += OUT_BYTES / 2;
            to_produce -= OUT_BYTES / 2;
        }

        buf_len = blake2b_init(h, to_produce, thr_id);
        buf_len = blake2b_update_local(out_buffer, OUT_BYTES, h, buf, buf_len, shfl, thr_id);
        blake2b_final_global(out, to_produce, h, buf, buf_len, shfl, thr_id);
    }
}

void blake2b_digestLong_local(__global uint *out, int out_len,
                        __local uint *in, int in_len,
                        int thr_id, __local ulong* shared)
{
    __local ulong *h = shared;
    __local ulong *shfl = &h[10];
    __local uint *buf = (__local uint *)&shfl[16];
    __local uint *out_buffer = &buf[32];
    int buf_len;

    if(thr_id == 0) buf[0] = (out_len * 4);
    buf_len = 1;

    if (out_len <= OUT_BYTES) {
        blake2b_init(h, out_len, thr_id);
        buf_len = blake2b_update_local(in, in_len, h, buf, buf_len, shfl, thr_id);
        blake2b_final_global(out, out_len, h, buf, buf_len, shfl, thr_id);
    } else {
        __local uint *cursor_in = out_buffer;
        __global uint *cursor_out = out;

        blake2b_init(h, OUT_BYTES, thr_id);
        buf_len = blake2b_update_local(in, in_len, h, buf, buf_len, shfl, thr_id);
        blake2b_final_local(out_buffer, OUT_BYTES, h, buf, buf_len, shfl, thr_id);

        for(int i=0; i < (OUT_BYTES / 8); i++, cursor_in += 4, cursor_out += 4) {
            cursor_out[thr_id] = cursor_in[thr_id];
        }

        out += OUT_BYTES / 2;

        int to_produce = out_len - OUT_BYTES / 2;
        while (to_produce > OUT_BYTES) {
            buf_len = blake2b_init(h, OUT_BYTES, thr_id);
            buf_len = blake2b_update_local(out_buffer, OUT_BYTES, h, buf, buf_len, shfl, thr_id);
            blake2b_final_local(out_buffer, OUT_BYTES, h, buf, buf_len, shfl, thr_id);

            cursor_out = out;
            cursor_in = out_buffer;
            for(int i=0; i < (OUT_BYTES / 8); i++, cursor_in += 4, cursor_out += 4) {
                cursor_out[thr_id] = cursor_in[thr_id];
            }

            out += OUT_BYTES / 2;
            to_produce -= OUT_BYTES / 2;
        }

        buf_len = blake2b_init(h, to_produce, thr_id);
        buf_len = blake2b_update_local(out_buffer, OUT_BYTES, h, buf, buf_len, shfl, thr_id);
        blake2b_final_global(out, to_produce, h, buf, buf_len, shfl, thr_id);
    }
}

#define fBlaMka(x, y) ((x) + (y) + 2 * upsample(mul_hi((uint)(x), (uint)(y)), (uint)(x) * (uint)y))

#define COMPUTE \
    a = fBlaMka(a, b);          \
    d = rotr64(d ^ a, (ulong)32);      \
    c = fBlaMka(c, d);          \
    b = rotr64(b ^ c, (ulong)24);      \
    a = fBlaMka(a, b);          \
    d = rotr64(d ^ a, (ulong)16);      \
    c = fBlaMka(c, d);          \
    b = rotr64(b ^ c, (ulong)63);

__constant char offsets_round_1[32][4] = {
        { 0, 4, 8, 12 },
        { 1, 5, 9, 13 },
        { 2, 6, 10, 14 },
        { 3, 7, 11, 15 },
        { 16, 20, 24, 28 },
        { 17, 21, 25, 29 },
        { 18, 22, 26, 30 },
        { 19, 23, 27, 31 },
        { 32, 36, 40, 44 },
        { 33, 37, 41, 45 },
        { 34, 38, 42, 46 },
        { 35, 39, 43, 47 },
        { 48, 52, 56, 60 },
        { 49, 53, 57, 61 },
        { 50, 54, 58, 62 },
        { 51, 55, 59, 63 },
        { 64, 68, 72, 76 },
        { 65, 69, 73, 77 },
        { 66, 70, 74, 78 },
        { 67, 71, 75, 79 },
        { 80, 84, 88, 92 },
        { 81, 85, 89, 93 },
        { 82, 86, 90, 94 },
        { 83, 87, 91, 95 },
        { 96, 100, 104, 108 },
        { 97, 101, 105, 109 },
        { 98, 102, 106, 110 },
        { 99, 103, 107, 111 },
        { 112, 116, 120, 124 },
        { 113, 117, 121, 125 },
        { 114, 118, 122, 126 },
        { 115, 119, 123, 127 },
};

__constant char offsets_round_2[32][4] = {
        { 0, 5, 10, 15 },
        { 1, 6, 11, 12 },
        { 2, 7, 8, 13 },
        { 3, 4, 9, 14 },
        { 16, 21, 26, 31 },
        { 17, 22, 27, 28 },
        { 18, 23, 24, 29 },
        { 19, 20, 25, 30 },
        { 32, 37, 42, 47 },
        { 33, 38, 43, 44 },
        { 34, 39, 40, 45 },
        { 35, 36, 41, 46 },
        { 48, 53, 58, 63 },
        { 49, 54, 59, 60 },
        { 50, 55, 56, 61 },
        { 51, 52, 57, 62 },
        { 64, 69, 74, 79 },
        { 65, 70, 75, 76 },
        { 66, 71, 72, 77 },
        { 67, 68, 73, 78 },
        { 80, 85, 90, 95 },
        { 81, 86, 91, 92 },
        { 82, 87, 88, 93 },
        { 83, 84, 89, 94 },
        { 96, 101, 106, 111 },
        { 97, 102, 107, 108 },
        { 98, 103, 104, 109 },
        { 99, 100, 105, 110 },
        { 112, 117, 122, 127 },
        { 113, 118, 123, 124 },
        { 114, 119, 120, 125 },
        { 115, 116, 121, 126 },
};

__constant char offsets_round_3[32][4] = {
        { 0, 32, 64, 96 },
        { 1, 33, 65, 97 },
        { 16, 48, 80, 112 },
        { 17, 49, 81, 113 },
        { 2, 34, 66, 98 },
        { 3, 35, 67, 99 },
        { 18, 50, 82, 114 },
        { 19, 51, 83, 115 },
        { 4, 36, 68, 100 },
        { 5, 37, 69, 101 },
        { 20, 52, 84, 116 },
        { 21, 53, 85, 117 },
        { 6, 38, 70, 102 },
        { 7, 39, 71, 103 },
        { 22, 54, 86, 118 },
        { 23, 55, 87, 119 },
        { 8, 40, 72, 104 },
        { 9, 41, 73, 105 },
        { 24, 56, 88, 120 },
        { 25, 57, 89, 121 },
        { 10, 42, 74, 106 },
        { 11, 43, 75, 107 },
        { 26, 58, 90, 122 },
        { 27, 59, 91, 123 },
        { 12, 44, 76, 108 },
        { 13, 45, 77, 109 },
        { 28, 60, 92, 124 },
        { 29, 61, 93, 125 },
        { 14, 46, 78, 110 },
        { 15, 47, 79, 111 },
        { 30, 62, 94, 126 },
        { 31, 63, 95, 127 },
};

__constant char offsets_round_4[32][4] = {
        { 0, 33, 80, 113 },
        { 1, 48, 81, 96 },
        { 16, 49, 64, 97 },
        { 17, 32, 65, 112 },
        { 2, 35, 82, 115 },
        { 3, 50, 83, 98 },
        { 18, 51, 66, 99 },
        { 19, 34, 67, 114 },
        { 4, 37, 84, 117 },
        { 5, 52, 85, 100 },
        { 20, 53, 68, 101 },
        { 21, 36, 69, 116 },
        { 6, 39, 86, 119 },
        { 7, 54, 87, 102 },
        { 22, 55, 70, 103 },
        { 23, 38, 71, 118 },
        { 8, 41, 88, 121 },
        { 9, 56, 89, 104 },
        { 24, 57, 72, 105 },
        { 25, 40, 73, 120 },
        { 10, 43, 90, 123 },
        { 11, 58, 91, 106 },
        { 26, 59, 74, 107 },
        { 27, 42, 75, 122 },
        { 12, 45, 92, 125 },
        { 13, 60, 93, 108 },
        { 28, 61, 76, 109 },
        { 29, 44, 77, 124 },
        { 14, 47, 94, 127 },
        { 15, 62, 95, 110 },
        { 30, 63, 78, 111 },
        { 31, 46, 79, 126 },
};

#define G1(data) \
{ \
	barrier(CLK_LOCAL_MEM_FENCE); \
	a = data[i1_0]; \
	b = data[i1_1]; \
	c = data[i1_2]; \
	d = data[i1_3]; \
	COMPUTE \
	data[i1_1] = b; \
    data[i1_2] = c; \
    data[i1_3] = d; \
    barrier(CLK_LOCAL_MEM_FENCE); \
}

#define G2(data) \
{ \
	b = data[i2_1]; \
	c = data[i2_2]; \
	d = data[i2_3]; \
	COMPUTE \
	data[i2_0] = a; \
	data[i2_1] = b; \
    data[i2_2] = c; \
    data[i2_3] = d; \
    barrier(CLK_LOCAL_MEM_FENCE); \
}

#define G3(data) \
{ \
	a = data[i3_0]; \
	b = data[i3_1]; \
	c = data[i3_2]; \
	d = data[i3_3]; \
	COMPUTE \
	data[i3_1] = b; \
    data[i3_2] = c; \
    data[i3_3] = d; \
    barrier(CLK_LOCAL_MEM_FENCE); \
}

#define G4(data) \
{ \
	b = data[i4_1]; \
	c = data[i4_2]; \
	d = data[i4_3]; \
	COMPUTE \
	data[i4_0] = a; \
	data[i4_1] = b; \
    data[i4_2] = c; \
    data[i4_3] = d; \
    barrier(CLK_LOCAL_MEM_FENCE); \
}

__kernel void fill_blocks(__global ulong *chunk_0,
						__global ulong *chunk_1,
						__global ulong *chunk_2,
						__global ulong *chunk_3,
						__global ulong *chunk_4,
						__global ulong *chunk_5,
						__global ulong *seed,
						__global ulong *out,
						__global uint *refs,
						__global uint *idxs,
						__global uint *segments,
                        int memsize,
                        int lanes,
                        int seg_length,
                        int seg_count,
						int threads_per_chunk,
                        int thread_idx,
                        __local ulong *scratchpad) { // lanes * BLOCK_SIZE_ULONG
    ulong4 tmp;
	ulong a, b, c, d;

	int hash_base = get_group_id(0) * 2;
	int mem_hash = hash_base + thread_idx;
	int local_id = get_local_id(0);

    int hash_idx = (local_id / THREADS_PER_LANE) % 2;
    int wave_id = local_id % (THREADS_PER_LANE * 2);
	int id = wave_id % THREADS_PER_LANE;
	int lane = local_id / (THREADS_PER_LANE * 2);
	int lane_length = seg_length * 4;

    int hash = hash_base + hash_idx;

	ulong chunks[6];
	chunks[0] = (ulong)chunk_0;
	chunks[1] = (ulong)chunk_1;
	chunks[2] = (ulong)chunk_2;
	chunks[3] = (ulong)chunk_3;
	chunks[4] = (ulong)chunk_4;
	chunks[5] = (ulong)chunk_5;
	int chunk_index = mem_hash / threads_per_chunk;
	int chunk_offset = mem_hash - chunk_index * threads_per_chunk;
	__global ulong *memory = (__global ulong *)chunks[chunk_index] + chunk_offset * (memsize / 8);

	int i1_0 = offsets_round_1[id][0];
	int i1_1 = offsets_round_1[id][1];
	int i1_2 = offsets_round_1[id][2];
	int i1_3 = offsets_round_1[id][3];

	int i2_0 = offsets_round_2[id][0];
	int i2_1 = offsets_round_2[id][1];
	int i2_2 = offsets_round_2[id][2];
	int i2_3 = offsets_round_2[id][3];

	int i3_0 = offsets_round_3[id][0];
	int i3_1 = offsets_round_3[id][1];
	int i3_2 = offsets_round_3[id][2];
	int i3_3 = offsets_round_3[id][3];

	int i4_0 = offsets_round_4[id][0];
	int i4_1 = offsets_round_4[id][1];
	int i4_2 = offsets_round_4[id][2];
	int i4_3 = offsets_round_4[id][3];

	__global ulong *seed_mem = seed + hash * lanes * 2 * BLOCK_SIZE_ULONG + lane * 2 * BLOCK_SIZE_ULONG;
	__global ulong *seed_dst = memory + (lane * lane_length * 2 + hash_idx) * BLOCK_SIZE_ULONG;

	vstore4(vload4(id, seed_mem), id, seed_dst);

	seed_mem += BLOCK_SIZE_ULONG;
	seed_dst += (2 * BLOCK_SIZE_ULONG);

	vstore4(vload4(id, seed_mem), id, seed_dst);

	__global ulong *next_block;
	__global ulong *prev_block;
    __global uint *seg_refs;
    __global uint *seg_idxs;

	__local ulong *state = scratchpad + (lane * 2 + hash_idx) * BLOCK_SIZE_ULONG;

	segments += (lane * 3);

	for(int s=0; s < (seg_count / lanes); s++) {
		int idx = ((s == 0) ? 2 : 0); // index for first slice in each lane is 2

		int with_xor = ((s >= 4) ? 1 : 0);
		int keep = 1;
		int slice = s % 4;
		int pass = s / 4;
		__global int *cur_seg = &segments[s * lanes * 3];

		int cur_idx = cur_seg[0];
        int prev_idx = cur_seg[1];
        int seg_type = cur_seg[2];
        int ref_idx = 0;

		prev_block = memory + prev_idx * 2 * BLOCK_SIZE_ULONG;

		tmp = vload4(wave_id, prev_block);

        if(seg_type == 0) {
            seg_refs = refs + ((s * lanes + lane) * seg_length - ((s > 0) ? lanes : lane) * 2);
            ref_idx = seg_refs[0];
            prefetch(memory + ref_idx * 2 * BLOCK_SIZE_ULONG, BLOCK_SIZE_ULONG);

            if(idxs != 0) {
                seg_idxs = idxs + ((s * lanes + lane) * seg_length - ((s > 0) ? lanes : lane) * 2);
                cur_idx = seg_idxs[0];
            }

            for (int i=0;idx < seg_length;i++, idx++) {
    			next_block = memory + (cur_idx & 0x7FFFFFFF) * 2 * BLOCK_SIZE_ULONG;

                if(with_xor == 1)
                    prefetch(next_block, BLOCK_SIZE_ULONG);

                tmp ^= vload4(wave_id, memory + ref_idx * 2 * BLOCK_SIZE_ULONG);

                if (idx < seg_length - 1) {
                    ref_idx = seg_refs[i + 1];
                    prefetch(memory + ref_idx * 2 * BLOCK_SIZE_ULONG, BLOCK_SIZE_ULONG);

                    if(idxs != 0) {
                        keep = cur_idx & 0x80000000;
                        cur_idx = seg_idxs[i + 1];
                    }
                    else
                        cur_idx++;
                }

                vstore4(tmp, id, state);

                G1(state);
                G2(state);
                G3(state);
                G4(state);

                if(with_xor == 1)
                    tmp ^= vload4(wave_id, next_block);

                tmp ^= vload4(id, state);

                if(keep > 0) {
                    vstore4(tmp, wave_id, next_block);
                    barrier(CLK_GLOBAL_MEM_FENCE);
                }
            }
        }
        else {
            vstore4(tmp, id, state);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int i=0;idx < seg_length;i++, idx++, cur_idx++) {
    			next_block = memory + cur_idx * 2 * BLOCK_SIZE_ULONG;

                if(with_xor == 1)
                    prefetch(next_block, BLOCK_SIZE_ULONG);

                ulong pseudo_rand = state[0];

                if(lanes == 1) {
                    uint reference_area_size = 0;

                    if(pass > 0) {
                        reference_area_size = lane_length - seg_length + idx - 1;
                    } else {
                        reference_area_size = slice * seg_length + idx - 1; // seg_length
                    }

                    ulong relative_position = pseudo_rand & 0xFFFFFFFF;
                    relative_position = (relative_position * relative_position) >> 32;

                    relative_position = reference_area_size - 1 -
                                        ((reference_area_size * relative_position) >> 32);

    				ref_idx = (((pass > 0 && slice < 3) ? ((slice + 1) * seg_length) : 0) + relative_position) % lane_length;
                }
                else {
                    ulong ref_lane = ((pseudo_rand >> 32)) % lanes; // thr_cost
                    uint reference_area_size = 0;

                    if(pass > 0) {
    					if (lane == ref_lane) {
                            reference_area_size = lane_length - seg_length + idx - 1;
    					} else {
    						reference_area_size = lane_length - seg_length + ((idx == 0) ? (-1) : 0);
    					}
                    }
                    else {
    					if (lane == ref_lane) {
                            reference_area_size = slice * seg_length + idx - 1; // seg_length
    					} else {
    						reference_area_size = slice * seg_length + ((idx == 0) ? (-1) : 0);
    					}
                    }

                    ulong relative_position = pseudo_rand & 0xFFFFFFFF;
                    relative_position = (relative_position * relative_position) >> 32;

                    relative_position = reference_area_size - 1 -
                                        ((reference_area_size * relative_position) >> 32);

    				ref_idx = ref_lane * lane_length + (((pass > 0 && slice < 3) ? ((slice + 1) * seg_length) : 0) + relative_position) % lane_length;
                }

                tmp ^= vload4(wave_id, memory + ref_idx * 2 * BLOCK_SIZE_ULONG);

                vstore4(tmp, id, state);

                G1(state);
                G2(state);
                G3(state);
                G4(state);

                if(with_xor == 1)
                    tmp ^= vload4(wave_id, next_block);

                tmp ^= vload4(id, state);

                vstore4(tmp, id, state);
                vstore4(tmp, wave_id, next_block);
                barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            }
        }
    }

    vstore4(tmp, id, state);
    barrier(CLK_LOCAL_MEM_FENCE);

	if(lane == 0) { // first lane needs to acumulate results
    	__global ulong *out_mem = out + hash * BLOCK_SIZE_ULONG;
		for(int l=1; l<lanes; l++)
            tmp ^= vload4(id, scratchpad + (l * 2 + hash_idx) * BLOCK_SIZE_ULONG);

        vstore4(tmp, id, out_mem);
	}
};

__kernel void prehash (
        __global uint *preseed,
        __global uint *seed,
		int memsz,
		int lanes,
		int passes,
		int pwdlen,
		int saltlen,
        int threads,
        __local ulong *blake_shared) {
	int seeds_batch_size = get_local_size(0) / 4; // number of seeds per block
	int hash_batch_size = seeds_batch_size / (lanes * 2); // number of hashes per block

	int id = get_local_id(0); // minimum 64 threads
	int thr_id = id % 4; // thread id in session
	int session = id / 4; // blake2b hashing session

    int hash = get_group_id(0) * hash_batch_size;
    int hash_idx = session / (lanes * 2);

    hash += hash_idx;

    if(hash < threads) {
        int hash_session = session % (lanes * 2); // session in hash

        int lane = hash_session / 2;  // 2 seeds per lane
        int idx = hash_session % 2; // seed idx in lane

        __local uint *local_mem = (__local uint *)&blake_shared[session * BLAKE_SHARED_MEM_ULONG];
        __global uint *local_seed = seed + (hash * lanes * 2 + hash_session) * BLOCK_SIZE_UINT;

        __local ulong *h = (__local ulong *)&local_mem[20];
        __local ulong *shfl = &h[10];
        __local uint *buf = (__local uint *)&shfl[16];
        __local uint *value = &buf[32];
        __local uint *local_preseed = &value[1];

        __global uint *cursor_in = preseed;
        __local uint *cursor_out = local_preseed;

        for(int i=0; i < (pwdlen >> 2); i++, cursor_in += 4, cursor_out += 4) {
            cursor_out[thr_id] = cursor_in[thr_id];
        }

        if(thr_id == 0) {
            for (int i = 0; i < (pwdlen % 4); i++) {
                cursor_out[i] = cursor_in[i];
            }

            uint nonce = (preseed[9] >> 24) | (preseed[10] << 8);
            nonce += hash;
            local_preseed[9] = (preseed[9] & 0x00FFFFFF) | (nonce << 24);
            local_preseed[10] = (preseed[10] & 0xFF000000) | (nonce >> 8);
        }

        int buf_len = blake2b_init(h, ARGON2_PREHASH_DIGEST_LENGTH_UINT, thr_id);
        *value = lanes; //lanes
        buf_len = blake2b_update_local(value, 1, h, buf, buf_len, shfl, thr_id);
        *value = 32; //outlen
        buf_len = blake2b_update_local(value, 1, h, buf, buf_len, shfl, thr_id);
        *value = memsz; //m_cost
        buf_len = blake2b_update_local(value, 1, h, buf, buf_len, shfl, thr_id);
        *value = passes; //t_cost
        buf_len = blake2b_update_local(value, 1, h, buf, buf_len, shfl, thr_id);
        *value = ARGON2_VERSION; //version
        buf_len = blake2b_update_local(value, 1, h, buf, buf_len, shfl, thr_id);
        *value = ARGON2_TYPE_VALUE; //type
        buf_len = blake2b_update_local(value, 1, h, buf, buf_len, shfl, thr_id);
        *value = pwdlen * 4; //pw_len
        buf_len = blake2b_update_local(value, 1, h, buf, buf_len, shfl, thr_id);
        buf_len = blake2b_update_local(local_preseed, pwdlen, h, buf, buf_len, shfl, thr_id);
        *value = saltlen * 4; //salt_len
        buf_len = blake2b_update_local(value, 1, h, buf, buf_len, shfl, thr_id);
        buf_len = blake2b_update_local(local_preseed, saltlen, h, buf, buf_len, shfl, thr_id);
        *value = 0; //secret_len
        buf_len = blake2b_update_local(value, 1, h, buf, buf_len, shfl, thr_id);
        buf_len = blake2b_update_local(0, 0, h, buf, buf_len, shfl, thr_id);
        *value = 0; //ad_len
        buf_len = blake2b_update_local(value, 1, h, buf, buf_len, shfl, thr_id);
        buf_len = blake2b_update_local(0, 0, h, buf, buf_len, shfl, thr_id);

        blake2b_final_local(local_mem, ARGON2_PREHASH_DIGEST_LENGTH_UINT, h, buf, buf_len, shfl, thr_id);

        if (thr_id == 0) {
            local_mem[ARGON2_PREHASH_DIGEST_LENGTH_UINT] = idx;
            local_mem[ARGON2_PREHASH_DIGEST_LENGTH_UINT + 1] = lane;
        }

        blake2b_digestLong_local(local_seed, ARGON2_DWORDS_IN_BLOCK, local_mem, ARGON2_PREHASH_SEED_LENGTH_UINT, thr_id, (__local ulong *)&local_mem[20]);
    }
}

__kernel void posthash (
        __global uint *hash,
        __global uint *out,
        __global uint *preseed,
        __local ulong *blake_shared) {

	int hash_id = get_group_id(0);
	int thread = get_local_id(0);

    __global uint *local_hash = hash + hash_id * (ARGON2_RAW_LENGTH + 1);
    __global uint *local_out = out + hash_id * BLOCK_SIZE_UINT;

    blake2b_digestLong_global(local_hash, ARGON2_RAW_LENGTH, local_out, ARGON2_DWORDS_IN_BLOCK, thread, blake_shared);
    if(thread == 0) {
        uint nonce = (preseed[9] >> 24) | (preseed[10] << 8);
        nonce += hash_id;
        local_hash[ARGON2_RAW_LENGTH] = nonce;
    }
}

)OCL";
