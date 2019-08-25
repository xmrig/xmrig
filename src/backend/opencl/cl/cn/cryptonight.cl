R"===(
/*
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
  */

/* For Mesa clover support */
#ifdef cl_clang_storage_class_specifiers
#   pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif

//#include "opencl/wolf-aes.cl"
XMRIG_INCLUDE_WOLF_AES
//#include "opencl/wolf-skein.cl"
XMRIG_INCLUDE_WOLF_SKEIN
//#include "opencl/jh.cl"
XMRIG_INCLUDE_JH
//#include "opencl/blake256.cl"
XMRIG_INCLUDE_BLAKE256
//#include "opencl/groestl256.cl"
XMRIG_INCLUDE_GROESTL256
//#include "fast_int_math_v2.cl"
XMRIG_INCLUDE_FAST_INT_MATH_V2
//#include "fast_div_heavy.cl"
XMRIG_INCLUDE_FAST_DIV_HEAVY


#define VARIANT_0    0  // Original CryptoNight or CryptoNight-Heavy
#define VARIANT_1    1  // CryptoNight variant 1 also known as Monero7 and CryptoNightV7
#define VARIANT_TUBE 2  // Modified CryptoNight Lite variant 1 with XOR (IPBC/TUBE only)
#define VARIANT_XTL  3  // Modified CryptoNight variant 1 (Stellite only)
#define VARIANT_MSR  4  // Modified CryptoNight variant 1 (Masari only)
#define VARIANT_XHV  5  // Modified CryptoNight-Heavy (Haven Protocol only)
#define VARIANT_XAO  6  // Modified CryptoNight variant 0 (Alloy only)
#define VARIANT_RTO  7  // Modified CryptoNight variant 1 (Arto only)
#define VARIANT_2    8  // CryptoNight variant 2
#define VARIANT_HALF 9  // CryptoNight variant 2 with half iterations (Masari/Stellite)
#define VARIANT_TRTL 10 // CryptoNight Turtle (TRTL)
#define VARIANT_GPU  11 // CryptoNight-GPU (Ryo)

#define CRYPTONIGHT       0 /* CryptoNight (2 MB) */
#define CRYPTONIGHT_LITE  1 /* CryptoNight (1 MB) */
#define CRYPTONIGHT_HEAVY 2 /* CryptoNight (4 MB) */
#define CRYPTONIGHT_PICO  3 /* CryptoNight (256 KB) */

#if defined(__NV_CL_C_VERSION) && STRIDED_INDEX != 0
#   undef STRIDED_INDEX
#   define STRIDED_INDEX 0
#endif


static const __constant ulong keccakf_rndc[24] =
{
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

static const __constant uchar sbox[256] =
{
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
};


void keccakf1600(ulong *s)
{
    for(int i = 0; i < 24; ++i)
    {
        ulong bc[5], tmp1, tmp2;
        bc[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20] ^ rotate(s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22], 1UL);
        bc[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21] ^ rotate(s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23], 1UL);
        bc[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22] ^ rotate(s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24], 1UL);
        bc[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23] ^ rotate(s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20], 1UL);
        bc[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24] ^ rotate(s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21], 1UL);

        tmp1 = s[1] ^ bc[0];

        s[0] ^= bc[4];
        s[1] = rotate(s[6] ^ bc[0], 44UL);
        s[6] = rotate(s[9] ^ bc[3], 20UL);
        s[9] = rotate(s[22] ^ bc[1], 61UL);
        s[22] = rotate(s[14] ^ bc[3], 39UL);
        s[14] = rotate(s[20] ^ bc[4], 18UL);
        s[20] = rotate(s[2] ^ bc[1], 62UL);
        s[2] = rotate(s[12] ^ bc[1], 43UL);
        s[12] = rotate(s[13] ^ bc[2], 25UL);
        s[13] = rotate(s[19] ^ bc[3], 8UL);
        s[19] = rotate(s[23] ^ bc[2], 56UL);
        s[23] = rotate(s[15] ^ bc[4], 41UL);
        s[15] = rotate(s[4] ^ bc[3], 27UL);
        s[4] = rotate(s[24] ^ bc[3], 14UL);
        s[24] = rotate(s[21] ^ bc[0], 2UL);
        s[21] = rotate(s[8] ^ bc[2], 55UL);
        s[8] = rotate(s[16] ^ bc[0], 35UL);
        s[16] = rotate(s[5] ^ bc[4], 36UL);
        s[5] = rotate(s[3] ^ bc[2], 28UL);
        s[3] = rotate(s[18] ^ bc[2], 21UL);
        s[18] = rotate(s[17] ^ bc[1], 15UL);
        s[17] = rotate(s[11] ^ bc[0], 10UL);
        s[11] = rotate(s[7] ^ bc[1], 6UL);
        s[7] = rotate(s[10] ^ bc[4], 3UL);
        s[10] = rotate(tmp1, 1UL);

        tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
        tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
        tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
        tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
        tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
        s[0] ^= keccakf_rndc[i];
    }
}

static const __constant uint keccakf_rotc[24] =
{
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

static const __constant uint keccakf_piln[24] =
{
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

void keccakf1600_1(ulong *st)
{
    int i, round;
    ulong t, bc[5];

    #pragma unroll 1
    for(round = 0; round < 24; ++round)
    {

        // Theta
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        #pragma unroll 1
        for (i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ rotate(bc[(i + 1) % 5], 1UL);
            st[i     ] ^= t;
            st[i +  5] ^= t;
            st[i + 10] ^= t;
            st[i + 15] ^= t;
            st[i + 20] ^= t;
        }

        // Rho Pi
        t = st[1];
        #pragma unroll
        for (i = 0; i < 24; ++i) {
            bc[0] = st[keccakf_piln[i]];
            st[keccakf_piln[i]] = rotate(t, (ulong)keccakf_rotc[i]);
            t = bc[0];
        }

        #pragma unroll 1
        for(int i = 0; i < 25; i += 5)
        {
            ulong tmp[5];

            #pragma unroll 1
            for(int x = 0; x < 5; ++x)
                tmp[x] = bitselect(st[i + x] ^ st[i + ((x + 2) % 5)], st[i + x], st[i + ((x + 1) % 5)]);

            #pragma unroll 1
            for(int x = 0; x < 5; ++x) st[i + x] = tmp[x];
        }

        //  Iota
        st[0] ^= keccakf_rndc[round];
    }
}

)==="
R"===(

void keccakf1600_2(__local ulong *st)
{
    int i, round;
    ulong t, bc[5];

    #pragma unroll 1
    for (round = 0; round < 24; ++round)
    {
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20] ^ rotate(st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22], 1UL);
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21] ^ rotate(st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23], 1UL);
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22] ^ rotate(st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24], 1UL);
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23] ^ rotate(st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20], 1UL);
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24] ^ rotate(st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21], 1UL);

        st[0] ^= bc[4];
        st[5] ^= bc[4];
        st[10] ^= bc[4];
        st[15] ^= bc[4];
        st[20] ^= bc[4];

        st[1] ^= bc[0];
        st[6] ^= bc[0];
        st[11] ^= bc[0];
        st[16] ^= bc[0];
        st[21] ^= bc[0];

        st[2] ^= bc[1];
        st[7] ^= bc[1];
        st[12] ^= bc[1];
        st[17] ^= bc[1];
        st[22] ^= bc[1];

        st[3] ^= bc[2];
        st[8] ^= bc[2];
        st[13] ^= bc[2];
        st[18] ^= bc[2];
        st[23] ^= bc[2];

        st[4] ^= bc[3];
        st[9] ^= bc[3];
        st[14] ^= bc[3];
        st[19] ^= bc[3];
        st[24] ^= bc[3];

        // Rho Pi
        t = st[1];
        #pragma unroll
        for (i = 0; i < 24; ++i) {
            bc[0] = st[keccakf_piln[i]];
            st[keccakf_piln[i]] = rotate(t, (ulong)keccakf_rotc[i]);
            t = bc[0];
        }

        #pragma unroll
        for(int i = 0; i < 25; i += 5)
        {
            ulong tmp1 = st[i], tmp2 = st[i + 1];

            st[i] = bitselect(st[i] ^ st[i + 2], st[i], st[i + 1]);
            st[i + 1] = bitselect(st[i + 1] ^ st[i + 3], st[i + 1], st[i + 2]);
            st[i + 2] = bitselect(st[i + 2] ^ st[i + 4], st[i + 2], st[i + 3]);
            st[i + 3] = bitselect(st[i + 3] ^ tmp1, st[i + 3], st[i + 4]);
            st[i + 4] = bitselect(st[i + 4] ^ tmp2, st[i + 4], tmp1);
        }

        //  Iota
        st[0] ^= keccakf_rndc[round];
    }
}

)==="
R"===(

void CNKeccak(ulong *output, ulong *input)
{
    ulong st[25];

    // Copy 72 bytes
    for(int i = 0; i < 9; ++i) st[i] = input[i];

    // Last four and '1' bit for padding
    //st[9] = as_ulong((uint2)(((uint *)input)[18], 0x00000001U));

    st[9] = (input[9] & 0x00000000FFFFFFFFUL) | 0x0000000100000000UL;

    for(int i = 10; i < 25; ++i) st[i] = 0x00UL;

    // Last bit of padding
    st[16] = 0x8000000000000000UL;

    keccakf1600_1(st);

    for(int i = 0; i < 25; ++i) output[i] = st[i];
}

static const __constant uchar rcon[8] = { 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40 };

#define SubWord(inw)        ((sbox[BYTE(inw, 3)] << 24) | (sbox[BYTE(inw, 2)] << 16) | (sbox[BYTE(inw, 1)] << 8) | sbox[BYTE(inw, 0)])

void AESExpandKey256(uint *keybuf)
{
    //#pragma unroll 4
    for(uint c = 8, i = 1; c < 40; ++c)
    {
        // For 256-bit keys, an sbox permutation is done every other 4th uint generated, AND every 8th
        uint t = ((!(c & 7)) || ((c & 7) == 4)) ? SubWord(keybuf[c - 1]) : keybuf[c - 1];

        // If the uint we're generating has an index that is a multiple of 8, rotate and XOR with the round constant,
        // then XOR this with previously generated uint. If it's 4 after a multiple of 8, only the sbox permutation
        // is done, followed by the XOR. If neither are true, only the XOR with the previously generated uint is done.
        keybuf[c] = keybuf[c - 8] ^ ((!(c & 7)) ? rotate(t, 24U) ^ as_uint((uchar4)(rcon[i++], 0U, 0U, 0U)) : t);
    }
}

#define MEM_CHUNK (1 << MEM_CHUNK_EXPONENT)

#if (STRIDED_INDEX == 0)
#   define IDX(x)   (x)
#elif (STRIDED_INDEX == 1)
#   if (ALGO == CRYPTONIGHT_HEAVY)
#       define IDX(x)   ((x) * WORKSIZE)
#   else
#       define IDX(x)   mul24((x), Threads)
#   endif
#elif (STRIDED_INDEX == 2)
#   define IDX(x)   (((x) % MEM_CHUNK) + ((x) / MEM_CHUNK) * WORKSIZE * MEM_CHUNK)
#endif

inline ulong getIdx()
{
#   if (STRIDED_INDEX == 0 || STRIDED_INDEX == 1 || STRIDED_INDEX == 2)
    return get_global_id(0) - get_global_offset(0);
#   endif
}

//#include "opencl/cryptonight_gpu.cl"
XMRIG_INCLUDE_CN_GPU

#define mix_and_propagate(xin) (xin)[(get_local_id(1)) % 8][get_local_id(0)] ^ (xin)[(get_local_id(1) + 1) % 8][get_local_id(0)]

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void cn0(__global ulong *input, __global uint4 *Scratchpad, __global ulong *states, uint Threads)
{
    uint ExpandedKey1[40];
    __local uint AES0[256], AES1[256], AES2[256], AES3[256];
    uint4 text;

    const uint gIdx = getIdx();

    for (int i = get_local_id(1) * 8 + get_local_id(0); i < 256; i += 8 * 8) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
        AES2[i] = rotate(tmp, 16U);
        AES3[i] = rotate(tmp, 24U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local ulong State_buf[8 * 25];

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;

#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO == CRYPTONIGHT_HEAVY)
            Scratchpad += (gIdx / WORKSIZE) * (MEMORY >> 4) * WORKSIZE + (gIdx % WORKSIZE);
#       else
            Scratchpad += gIdx;
#       endif
#       elif (STRIDED_INDEX == 2)
        Scratchpad += (gIdx / WORKSIZE) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * (gIdx % WORKSIZE);
#       endif

        if (get_local_id(1) == 0)
        {
            __local ulong* State = State_buf + get_local_id(0) * 25;

            ((__local ulong8 *)State)[0] = vload8(0, input);
            State[8]  = input[8];
            State[9]  = input[9];
            State[10] = input[10];
            State[11] = input[11];
            State[12] = input[12];
            State[13] = input[13];
            State[14] = input[14];
            State[15] = input[15];

            ((__local uint *)State)[9]  &= 0x00FFFFFFU;
            ((__local uint *)State)[9]  |= (((uint)get_global_id(0)) & 0xFF) << 24;
            ((__local uint *)State)[10] &= 0xFF000000U;
            /* explicit cast to `uint` is required because some OpenCL implementations (e.g. NVIDIA)
             * handle get_global_id and get_global_offset as signed long long int and add
             * 0xFFFFFFFF... to `get_global_id` if we set on host side a 32bit offset where the first bit is `1`
             * (even if it is correct casted to unsigned on the host)
             */
            ((__local uint *)State)[10] |= (((uint)get_global_id(0) >> 8));

            // Last bit of padding
            State[16] = 0x8000000000000000UL;

            for (int i = 17; i < 25; ++i) {
                State[i] = 0x00UL;
            }

            keccakf1600_2(State);

            #pragma unroll
            for (int i = 0; i < 25; ++i) {
                states[i] = State[i];
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        text = vload4(get_local_id(1) + 4, (__global uint *)(states));

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            ((ulong *)ExpandedKey1)[i] = states[i];
        }

        AESExpandKey256(ExpandedKey1);
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

#   if (ALGO == CRYPTONIGHT_HEAVY)
    {
        __local uint4 xin[8][8];

        /* Also left over threads perform this loop.
         * The left over thread results will be ignored
         */
        #pragma unroll 16
        for (size_t i = 0; i < 16; i++) {
            #pragma unroll 10
            for (int j = 0; j < 10; ++j) {
                uint4 t = ((uint4 *)ExpandedKey1)[j];
                t.s0 ^= AES0[BYTE(text.s0, 0)] ^ AES1[BYTE(text.s1, 1)] ^ AES2[BYTE(text.s2, 2)] ^ AES3[BYTE(text.s3, 3)];
                t.s1 ^= AES0[BYTE(text.s1, 0)] ^ AES1[BYTE(text.s2, 1)] ^ AES2[BYTE(text.s3, 2)] ^ AES3[BYTE(text.s0, 3)];
                t.s2 ^= AES0[BYTE(text.s2, 0)] ^ AES1[BYTE(text.s3, 1)] ^ AES2[BYTE(text.s0, 2)] ^ AES3[BYTE(text.s1, 3)];
                t.s3 ^= AES0[BYTE(text.s3, 0)] ^ AES1[BYTE(text.s0, 1)] ^ AES2[BYTE(text.s1, 2)] ^ AES3[BYTE(text.s2, 3)];
                text = t;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            xin[get_local_id(1)][get_local_id(0)] = text;
            barrier(CLK_LOCAL_MEM_FENCE);
            text = mix_and_propagate(xin);
        }
    }
#   endif

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        const uint local_id1 = get_local_id(1);
        #pragma unroll 2
        for(uint i = 0; i < (MEMORY >> 4); i += 8) {
            #pragma unroll 10
            for (uint j = 0; j < 10; ++j) {
                uint4 t = ((uint4 *)ExpandedKey1)[j];
                t.s0 ^= AES0[BYTE(text.s0, 0)] ^ AES1[BYTE(text.s1, 1)] ^ AES2[BYTE(text.s2, 2)] ^ AES3[BYTE(text.s3, 3)];
                t.s1 ^= AES0[BYTE(text.s1, 0)] ^ AES1[BYTE(text.s2, 1)] ^ AES2[BYTE(text.s3, 2)] ^ AES3[BYTE(text.s0, 3)];
                t.s2 ^= AES0[BYTE(text.s2, 0)] ^ AES1[BYTE(text.s3, 1)] ^ AES2[BYTE(text.s0, 2)] ^ AES3[BYTE(text.s1, 3)];
                t.s3 ^= AES0[BYTE(text.s3, 0)] ^ AES1[BYTE(text.s0, 1)] ^ AES2[BYTE(text.s1, 2)] ^ AES3[BYTE(text.s2, 3)];
                text = t;
            }

            Scratchpad[IDX(i + local_id1)] = text;
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}

)==="
R"===(

#define VARIANT1_1(p) \
        uint table = 0x75310U; \
        uint index = (((p).s2 >> 26) & 12) | (((p).s2 >> 23) & 2); \
        (p).s2 ^= ((table >> index) & 0x30U) << 24

#define VARIANT1_1_XTL(p) \
        uint table  = 0x75310U; \
        uint offset = variant == VARIANT_XTL ? 27 : 26; \
        uint index  = (((p).s2 >> offset) & 12) | (((p).s2 >> 23) & 2); \
        (p).s2 ^= ((table >> index) & 0x30U) << 24

#define VARIANT1_2(p) ((uint2 *)&(p))[0] ^= tweak1_2_0

#define VARIANT1_INIT() \
        tweak1_2 = as_uint2(input[4]); \
        tweak1_2.s0 >>= 24; \
        tweak1_2.s0 |= tweak1_2.s1 << 8; \
        tweak1_2.s1 = (uint) get_global_id(0); \
        tweak1_2 ^= as_uint2(states[24])

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1_monero(__global uint4 *Scratchpad, __global ulong *states, uint variant, __global ulong *input, uint Threads)
{
    ulong a[2], b[2];
    __local uint AES0[256], AES1[256];

    const ulong gIdx = getIdx();

    for (int i = get_local_id(0); i < 256; i += WORKSIZE) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint2 tweak1_2;
    uint4 b_x;
#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;
#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO == CRYPTONIGHT_HEAVY)
            Scratchpad += (gIdx / WORKSIZE) * (MEMORY >> 4) * WORKSIZE + (gIdx % WORKSIZE);
#       else
            Scratchpad += gIdx;
#       endif
#       elif (STRIDED_INDEX == 2)
        Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#       endif

        a[0] = states[0] ^ states[4];
        b[0] = states[2] ^ states[6];
        a[1] = states[1] ^ states[5];
        b[1] = states[3] ^ states[7];

        b_x = ((uint4 *)b)[0];
        VARIANT1_INIT();
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        #pragma unroll UNROLL_FACTOR
        for (int i = 0; i < ITERATIONS; ++i) {
            ulong c[2];

            ((uint4 *)c)[0] = Scratchpad[IDX((as_uint2(a[0]).s0 & MASK) >> 4)];
            ((uint4 *)c)[0] = AES_Round_Two_Tables(AES0, AES1, ((uint4 *)c)[0], ((uint4 *)a)[0]);

            b_x ^= ((uint4 *)c)[0];
            VARIANT1_1_XTL(b_x);
            Scratchpad[IDX((as_uint2(a[0]).s0 & MASK) >> 4)] = b_x;

            uint4 tmp;
            tmp = Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)];

            a[1] += c[0] * as_ulong2(tmp).s0;
            a[0] += mul_hi(c[0], as_ulong2(tmp).s0);

            uint2 tweak1_2_0 = tweak1_2;
            if (variant == VARIANT_RTO) {
                tweak1_2_0 ^= ((uint2 *)&(a[0]))[0];
            }

            VARIANT1_2(a[1]);
            Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)] = ((uint4 *)a)[0];
            VARIANT1_2(a[1]);

            ((uint4 *)a)[0] ^= tmp;

            b_x = ((uint4 *)c)[0];
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}


)==="
R"===(

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1_v2_monero(__global uint4 *Scratchpad, __global ulong *states, uint variant, __global ulong *input, uint Threads)
{
#   if (ALGO == CRYPTONIGHT || ALGO == CRYPTONIGHT_PICO)
    ulong a[2], b[4];
    __local uint AES0[256], AES1[256], AES2[256], AES3[256];
    
    const ulong gIdx = getIdx();

    for(int i = get_local_id(0); i < 256; i += WORKSIZE)
    {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
        AES2[i] = rotate(tmp, 16U);
        AES3[i] = rotate(tmp, 24U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;

#       if defined(__NV_CL_C_VERSION)
            Scratchpad += gIdx * (ITERATIONS >> 2);
#       else
#           if (STRIDED_INDEX == 0)
                Scratchpad += gIdx * (MEMORY >> 4);
#           elif (STRIDED_INDEX == 1)
                Scratchpad += gIdx;
#           elif (STRIDED_INDEX == 2)
                Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#           endif
#       endif

        a[0] = states[0] ^ states[4];
        a[1] = states[1] ^ states[5];

        b[0] = states[2] ^ states[6];
        b[1] = states[3] ^ states[7];
        b[2] = states[8] ^ states[10];
        b[3] = states[9] ^ states[11];
    }
    
    ulong2 bx0 = ((ulong2 *)b)[0];
    ulong2 bx1 = ((ulong2 *)b)[1];
    
    mem_fence(CLK_LOCAL_MEM_FENCE);

#   ifdef __NV_CL_C_VERSION
        __local uint16 scratchpad_line_buf[WORKSIZE];
        __local uint16* scratchpad_line = scratchpad_line_buf + get_local_id(0);
#       define SCRATCHPAD_CHUNK(N) (*(__local uint4*)((__local uchar*)(scratchpad_line) + (idx1 ^ (N << 4))))
#   else
#       if (STRIDED_INDEX == 0)
#           define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + (idx ^ (N << 4))))
#       elif (STRIDED_INDEX == 1)
#           define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + mul24(as_uint(idx ^ (N << 4)), Threads)))
#       elif (STRIDED_INDEX == 2)
#           define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + (((idx ^ (N << 4)) % (MEM_CHUNK << 4)) + ((idx ^ (N << 4)) / (MEM_CHUNK << 4)) * WORKSIZE * (MEM_CHUNK << 4))))
#       endif
#   endif

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
    uint2 division_result = as_uint2(states[12]);
    uint sqrt_result = as_uint2(states[13]).s0;

    #pragma unroll UNROLL_FACTOR
    for(int i = 0; i < ITERATIONS; ++i)
    {
#       ifdef __NV_CL_C_VERSION
            uint idx  = a[0] & 0x1FFFC0;
            uint idx1 = a[0] & 0x30;

            *scratchpad_line = *(__global uint16*)((__global uchar*)(Scratchpad) + idx);
#       else
            uint idx = a[0] & MASK;
#       endif

        uint4 c = SCRATCHPAD_CHUNK(0);
        c = AES_Round(AES0, AES1, AES2, AES3, c, ((uint4 *)a)[0]);

        {
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1));
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(3));

            SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
        }

        SCRATCHPAD_CHUNK(0) = as_uint4(bx0) ^ c;

#       ifdef __NV_CL_C_VERSION
            *(__global uint16*)((__global uchar*)(Scratchpad) + idx) = *scratchpad_line;

            idx = as_ulong2(c).s0 & 0x1FFFC0;
            idx1 = as_ulong2(c).s0 & 0x30;

            *scratchpad_line = *(__global uint16*)((__global uchar*)(Scratchpad) + idx);
#       else
            idx = as_ulong2(c).s0 & MASK;
#       endif

        uint4 tmp = SCRATCHPAD_CHUNK(0);

        {
            tmp.s0 ^= division_result.s0;
            tmp.s1 ^= division_result.s1 ^ sqrt_result;

            division_result = fast_div_v2(as_ulong2(c).s1, (c.s0 + (sqrt_result << 1)) | 0x80000001UL);
            sqrt_result = fast_sqrt_v2(as_ulong2(c).s0 + as_ulong(division_result));
        }

        ulong2 t;
        t.s0 = mul_hi(as_ulong2(c).s0, as_ulong2(tmp).s0);
        t.s1 = as_ulong2(c).s0 * as_ulong2(tmp).s0;
        {
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1)) ^ t;
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            t ^= chunk2;
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(3));

            SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
        }

        a[1] += t.s1;
        a[0] += t.s0;

        SCRATCHPAD_CHUNK(0) = ((uint4 *)a)[0];

#       ifdef __NV_CL_C_VERSION
            *(__global uint16*)((__global uchar*)(Scratchpad) + idx) = *scratchpad_line;
#       endif

        ((uint4 *)a)[0] ^= tmp;
        bx1 = bx0;
        bx0 = as_ulong2(c);
    }
    
#   undef SCRATCHPAD_CHUNK
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
#   endif
}

)==="
R"===(

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1_v2_half(__global uint4 *Scratchpad, __global ulong *states, uint variant, __global ulong *input, uint Threads)
{
#   if (ALGO == CRYPTONIGHT)
    ulong a[2], b[4];
    __local uint AES0[256], AES1[256], AES2[256], AES3[256];

    const ulong gIdx = getIdx();

    for(int i = get_local_id(0); i < 256; i += WORKSIZE)
    {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
        AES2[i] = rotate(tmp, 16U);
        AES3[i] = rotate(tmp, 24U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;

#       if defined(__NV_CL_C_VERSION)
            Scratchpad += gIdx * (0x40000 >> 2);
#       else
#           if (STRIDED_INDEX == 0)
                Scratchpad += gIdx * (MEMORY >> 4);
#           elif (STRIDED_INDEX == 1)
                Scratchpad += gIdx;
#           elif (STRIDED_INDEX == 2)
                Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#           endif
#       endif

        a[0] = states[0] ^ states[4];
        a[1] = states[1] ^ states[5];

        b[0] = states[2] ^ states[6];
        b[1] = states[3] ^ states[7];
        b[2] = states[8] ^ states[10];
        b[3] = states[9] ^ states[11];
    }

    ulong2 bx0 = ((ulong2 *)b)[0];
    ulong2 bx1 = ((ulong2 *)b)[1];

    mem_fence(CLK_LOCAL_MEM_FENCE);

#   ifdef __NV_CL_C_VERSION
        __local uint16 scratchpad_line_buf[WORKSIZE];
        __local uint16* scratchpad_line = scratchpad_line_buf + get_local_id(0);
#       define SCRATCHPAD_CHUNK(N) (*(__local uint4*)((__local uchar*)(scratchpad_line) + (idx1 ^ (N << 4))))
#   else
#       if (STRIDED_INDEX == 0)
#           define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + (idx ^ (N << 4))))
#       elif (STRIDED_INDEX == 1)
#           define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + mul24(as_uint(idx ^ (N << 4)), Threads)))
#       elif (STRIDED_INDEX == 2)
#           define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + (((idx ^ (N << 4)) % (MEM_CHUNK << 4)) + ((idx ^ (N << 4)) / (MEM_CHUNK << 4)) * WORKSIZE * (MEM_CHUNK << 4))))
#       endif
#   endif

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
    uint2 division_result = as_uint2(states[12]);
    uint sqrt_result = as_uint2(states[13]).s0;

    #pragma unroll UNROLL_FACTOR
    for(int i = 0; i < 0x40000; ++i)
    {
#       ifdef __NV_CL_C_VERSION
            uint idx  = a[0] & 0x1FFFC0;
            uint idx1 = a[0] & 0x30;

            *scratchpad_line = *(__global uint16*)((__global uchar*)(Scratchpad) + idx);
#       else
            uint idx = a[0] & MASK;
#       endif

        uint4 c = SCRATCHPAD_CHUNK(0);
        c = AES_Round(AES0, AES1, AES2, AES3, c, ((uint4 *)a)[0]);

        {
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1));
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(3));

            SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
        }

        SCRATCHPAD_CHUNK(0) = as_uint4(bx0) ^ c;

#       ifdef __NV_CL_C_VERSION
            *(__global uint16*)((__global uchar*)(Scratchpad) + idx) = *scratchpad_line;

            idx = as_ulong2(c).s0 & 0x1FFFC0;
            idx1 = as_ulong2(c).s0 & 0x30;

            *scratchpad_line = *(__global uint16*)((__global uchar*)(Scratchpad) + idx);
#       else
            idx = as_ulong2(c).s0 & MASK;
#       endif

        uint4 tmp = SCRATCHPAD_CHUNK(0);

        {
            tmp.s0 ^= division_result.s0;
            tmp.s1 ^= division_result.s1 ^ sqrt_result;

            division_result = fast_div_v2(as_ulong2(c).s1, (c.s0 + (sqrt_result << 1)) | 0x80000001UL);
            sqrt_result = fast_sqrt_v2(as_ulong2(c).s0 + as_ulong(division_result));
        }

        ulong2 t;
        t.s0 = mul_hi(as_ulong2(c).s0, as_ulong2(tmp).s0);
        t.s1 = as_ulong2(c).s0 * as_ulong2(tmp).s0;
        {
            const ulong2 chunk1 = as_ulong2(SCRATCHPAD_CHUNK(1)) ^ t;
            const ulong2 chunk2 = as_ulong2(SCRATCHPAD_CHUNK(2));
            t ^= chunk2;
            const ulong2 chunk3 = as_ulong2(SCRATCHPAD_CHUNK(3));

            SCRATCHPAD_CHUNK(1) = as_uint4(chunk3 + bx1);
            SCRATCHPAD_CHUNK(2) = as_uint4(chunk1 + bx0);
            SCRATCHPAD_CHUNK(3) = as_uint4(chunk2 + ((ulong2 *)a)[0]);
        }

        a[1] += t.s1;
        a[0] += t.s0;

        SCRATCHPAD_CHUNK(0) = ((uint4 *)a)[0];

#       ifdef __NV_CL_C_VERSION
            *(__global uint16*)((__global uchar*)(Scratchpad) + idx) = *scratchpad_line;
#       endif

        ((uint4 *)a)[0] ^= tmp;
        bx1 = bx0;
        bx0 = as_ulong2(c);
    }

#   undef SCRATCHPAD_CHUNK
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
#   endif
}

)==="
R"===(

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1_msr(__global uint4 *Scratchpad, __global ulong *states, uint variant, __global ulong *input, uint Threads)
{
#   if (ALGO == CRYPTONIGHT)
    ulong a[2], b[2];
    __local uint AES0[256], AES1[256];

    const ulong gIdx = getIdx();

    for (int i = get_local_id(0); i < 256; i += WORKSIZE) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint2 tweak1_2;
    uint4 b_x;
#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;
#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO == CRYPTONIGHT_HEAVY)
            Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + get_local_id(0);
#       else
            Scratchpad += gIdx;
#       endif
#       elif (STRIDED_INDEX == 2)
        Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#       endif

        a[0] = states[0] ^ states[4];
        b[0] = states[2] ^ states[6];
        a[1] = states[1] ^ states[5];
        b[1] = states[3] ^ states[7];

        b_x = ((uint4 *)b)[0];
        VARIANT1_INIT();
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        #pragma unroll 8
        for (int i = 0; i < 0x40000; ++i) {
            ulong c[2];

            ((uint4 *)c)[0] = Scratchpad[IDX((as_uint2(a[0]).s0 & MASK) >> 4)];
            ((uint4 *)c)[0] = AES_Round_Two_Tables(AES0, AES1, ((uint4 *)c)[0], ((uint4 *)a)[0]);

            b_x ^= ((uint4 *)c)[0];
            VARIANT1_1(b_x);
            Scratchpad[IDX((as_uint2(a[0]).s0 & MASK) >> 4)] = b_x;

            uint4 tmp;
            tmp = Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)];

            a[1] += c[0] * as_ulong2(tmp).s0;
            a[0] += mul_hi(c[0], as_ulong2(tmp).s0);

            uint2 tweak1_2_0 = tweak1_2;

            VARIANT1_2(a[1]);
            Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)] = ((uint4 *)a)[0];
            VARIANT1_2(a[1]);

            ((uint4 *)a)[0] ^= tmp;

            b_x = ((uint4 *)c)[0];
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
#   endif
}

)==="
R"===(

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1_tube(__global uint4 *Scratchpad, __global ulong *states, uint variant, __global ulong *input, uint Threads)
{
#   if (ALGO == CRYPTONIGHT_HEAVY)
    ulong a[2], b[2];
    __local uint AES0[256], AES1[256];

    const ulong gIdx = getIdx();

    for (int i = get_local_id(0); i < 256; i += WORKSIZE) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint2 tweak1_2;
    uint4 b_x;
#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;
#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO == CRYPTONIGHT_HEAVY)
            Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + get_local_id(0);
#       else
            Scratchpad += gIdx;
#       endif
#       elif (STRIDED_INDEX == 2)
        Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#       endif

        a[0] = states[0] ^ states[4];
        b[0] = states[2] ^ states[6];
        a[1] = states[1] ^ states[5];
        b[1] = states[3] ^ states[7];

        b_x = ((uint4 *)b)[0];
        VARIANT1_INIT();
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        uint idx0 = a[0];

        #pragma unroll UNROLL_FACTOR
        for (int i = 0; i < ITERATIONS; ++i) {
            ulong c[2];

            ((uint4 *)c)[0] = Scratchpad[IDX((idx0 & MASK) >> 4)];
            ((uint4 *)c)[0] = AES_Round_bittube2(AES0, AES1, ((uint4 *)c)[0], ((uint4 *)a)[0]);

            b_x ^= ((uint4 *)c)[0];
            VARIANT1_1(b_x);
            Scratchpad[IDX((idx0 & MASK) >> 4)] = b_x;

            uint4 tmp;
            tmp = Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)];

            a[1] += c[0] * as_ulong2(tmp).s0;
            a[0] += mul_hi(c[0], as_ulong2(tmp).s0);

            uint2 tweak1_2_0 = tweak1_2;
            tweak1_2_0 ^= ((uint2 *)&(a[0]))[0];

            VARIANT1_2(a[1]);
            Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)] = ((uint4 *)a)[0];
            VARIANT1_2(a[1]);

            ((uint4 *)a)[0] ^= tmp;
            idx0 = a[0];

            b_x = ((uint4 *)c)[0];

            {
                long n = *((__global long*)(Scratchpad + (IDX((idx0 & MASK) >> 4))));
                int d  = ((__global int*)(Scratchpad + (IDX((idx0 & MASK) >> 4))))[2];
                long q = fast_div_heavy(n, d | 0x5);
                *((__global long*)(Scratchpad + (IDX((idx0 & MASK) >> 4)))) = n ^ q;
                idx0 = d ^ q;
            }
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
#   endif
}

)==="
R"===(

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1(__global uint4 *Scratchpad, __global ulong *states, uint variant, __global ulong *input, uint Threads)
{
    ulong a[2], b[2];
    __local uint AES0[256], AES1[256];

    const ulong gIdx = getIdx();

    for (int i = get_local_id(0); i < 256; i += WORKSIZE) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint4 b_x;
#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;
#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO == CRYPTONIGHT_HEAVY)
            Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + get_local_id(0);
#       else
            Scratchpad += gIdx;
#       endif
#       elif(STRIDED_INDEX == 2)
        Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#       endif

        a[0] = states[0] ^ states[4];
        b[0] = states[2] ^ states[6];
        a[1] = states[1] ^ states[5];
        b[1] = states[3] ^ states[7];

        b_x = ((uint4 *)b)[0];
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        uint idx0 = a[0];

        #pragma unroll UNROLL_FACTOR
        for (int i = 0; i < ITERATIONS; ++i) {
            ulong c[2];

            ((uint4 *)c)[0] = Scratchpad[IDX((idx0 & MASK) >> 4)];
            ((uint4 *)c)[0] = AES_Round_Two_Tables(AES0, AES1, ((uint4 *)c)[0], ((uint4 *)a)[0]);

            Scratchpad[IDX((idx0 & MASK) >> 4)] = b_x ^ ((uint4 *)c)[0];

            uint4 tmp;
            tmp = Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)];

            a[1] += c[0] * as_ulong2(tmp).s0;
            a[0] += mul_hi(c[0], as_ulong2(tmp).s0);

            Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)] = ((uint4 *)a)[0];

            ((uint4 *)a)[0] ^= tmp;
            idx0 = a[0];

            b_x = ((uint4 *)c)[0];

#           if (ALGO == CRYPTONIGHT_HEAVY)
            {
                const long2 n = *((__global long2*)(Scratchpad + (IDX((idx0 & MASK) >> 4))));
                long q = fast_div_heavy(n.s0, as_int4(n).s2 | 0x5);
                *((__global long*)(Scratchpad + (IDX((idx0 & MASK) >> 4)))) = n.s0 ^ q;

                if (variant == VARIANT_XHV) {
                    idx0 = (~as_int4(n).s2) ^ q;
                } else {
                    idx0 = as_int4(n).s2 ^ q;
                }
            }
#           endif
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}

)==="
R"===(

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void cn1_xao(__global uint4 *Scratchpad, __global ulong *states, uint variant, __global ulong *input, uint Threads)
{
#   if (ALGO == CRYPTONIGHT)
    ulong a[2], b[2];
    __local uint AES0[256], AES1[256];

    const ulong gIdx = getIdx();

    for (int i = get_local_id(0); i < 256; i += WORKSIZE) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint4 b_x;
#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;
#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO == CRYPTONIGHT_HEAVY)
            Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + get_local_id(0);
#       else
            Scratchpad += gIdx;
#       endif
#       elif(STRIDED_INDEX == 2)
        Scratchpad += get_group_id(0) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * get_local_id(0);
#       endif

        a[0] = states[0] ^ states[4];
        b[0] = states[2] ^ states[6];
        a[1] = states[1] ^ states[5];
        b[1] = states[3] ^ states[7];

        b_x = ((uint4 *)b)[0];
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        uint idx0 = a[0];

        #pragma unroll 8
        for (int i = 0; i < 0x100000; ++i) {
            ulong c[2];

            ((uint4 *)c)[0] = Scratchpad[IDX((idx0 & MASK) >> 4)];
            ((uint4 *)c)[0] = AES_Round_Two_Tables(AES0, AES1, ((uint4 *)c)[0], ((uint4 *)a)[0]);

            Scratchpad[IDX((idx0 & MASK) >> 4)] = b_x ^ ((uint4 *)c)[0];

            uint4 tmp;
            tmp = Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)];

            a[1] += c[0] * as_ulong2(tmp).s0;
            a[0] += mul_hi(c[0], as_ulong2(tmp).s0);

            Scratchpad[IDX((as_uint2(c[0]).s0 & MASK) >> 4)] = ((uint4 *)a)[0];

            ((uint4 *)a)[0] ^= tmp;
            idx0 = a[0];

            b_x = ((uint4 *)c)[0];
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
#   endif
}

)==="
R"===(

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void cn2(__global uint4 *Scratchpad, __global ulong *states, __global uint *Branch0, __global uint *Branch1, __global uint *Branch2, __global uint *Branch3, uint Threads)
{
    __local uint AES0[256], AES1[256], AES2[256], AES3[256];
    uint ExpandedKey2[40];
    uint4 text;

    const ulong gIdx = getIdx();

    for (int i = get_local_id(1) * 8 + get_local_id(0); i < 256; i += 8 * 8) {
        const uint tmp = AES0_C[i];
        AES0[i] = tmp;
        AES1[i] = rotate(tmp, 8U);
        AES2[i] = rotate(tmp, 16U);
        AES3[i] = rotate(tmp, 24U);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        states += 25 * gIdx;
#       if (STRIDED_INDEX == 0)
        Scratchpad += gIdx * (MEMORY >> 4);
#       elif (STRIDED_INDEX == 1)
#       if (ALGO == CRYPTONIGHT_HEAVY)
            Scratchpad += (gIdx / WORKSIZE) * (MEMORY >> 4) * WORKSIZE + (gIdx % WORKSIZE);
#       else
            Scratchpad += gIdx;
#       endif
#       elif (STRIDED_INDEX == 2)
        Scratchpad += (gIdx / WORKSIZE) * (MEMORY >> 4) * WORKSIZE + MEM_CHUNK * (gIdx % WORKSIZE);
#       endif

        #if defined(__Tahiti__) || defined(__Pitcairn__)

        for(int i = 0; i < 4; ++i) ((ulong *)ExpandedKey2)[i] = states[i + 4];
        text = vload4(get_local_id(1) + 4, (__global uint *)states);

        #else

        text = vload4(get_local_id(1) + 4, (__global uint *)states);
        ((uint8 *)ExpandedKey2)[0] = vload8(1, (__global uint *)states);

        #endif

        AESExpandKey256(ExpandedKey2);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#   if (ALGO == CRYPTONIGHT_HEAVY)
    __local uint4 xin1[8][8];
    __local uint4 xin2[8][8];
    __local uint4* xin1_store = &xin1[get_local_id(1)][get_local_id(0)];
    __local uint4* xin1_load = &xin1[(get_local_id(1) + 1) % 8][get_local_id(0)];
    __local uint4* xin2_store = &xin2[get_local_id(1)][get_local_id(0)];
    __local uint4* xin2_load = &xin2[(get_local_id(1) + 1) % 8][get_local_id(0)];
    *xin2_store = (uint4)(0, 0, 0, 0);
#   endif

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
#       if (ALGO == CRYPTONIGHT_HEAVY)
        #pragma unroll 2
        for(int i = 0, i1 = get_local_id(1); i < (MEMORY >> 7); ++i, i1 = (i1 + 16) % (MEMORY >> 4))
        {
            text ^= Scratchpad[IDX(i1)];
            barrier(CLK_LOCAL_MEM_FENCE);
            text ^= *xin2_load;

            #pragma unroll 10
            for(int j = 0; j < 10; ++j)
                text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);

            *xin1_store = text;

            text ^= Scratchpad[IDX(i1 + 8)];
            barrier(CLK_LOCAL_MEM_FENCE);
            text ^= *xin1_load;

            #pragma unroll 10
            for(int j = 0; j < 10; ++j)
                text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);

            *xin2_store = text;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        text ^= *xin2_load;

#       else
        const uint local_id1 = get_local_id(1);
        #pragma unroll 2
        for (uint i = 0; i < (MEMORY >> 7); ++i) {
            text ^= Scratchpad[IDX((i << 3) + local_id1)];

            #pragma unroll 10
            for(uint j = 0; j < 10; ++j)
                text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);
        }
#       endif
    }

#   if (ALGO == CRYPTONIGHT_HEAVY)
    /* Also left over threads performe this loop.
     * The left over thread results will be ignored
     */
    #pragma unroll 16
    for(size_t i = 0; i < 16; i++)
    {
        #pragma unroll 10
        for (int j = 0; j < 10; ++j) {
            text = AES_Round(AES0, AES1, AES2, AES3, text, ((uint4 *)ExpandedKey2)[j]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        *xin1_store = text;
        barrier(CLK_LOCAL_MEM_FENCE);
        text ^= *xin1_load;
    }
#   endif

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        vstore2(as_ulong2(text), get_local_id(1) + 4, states);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    __local ulong State_buf[8 * 25];

#   if (COMP_MODE == 1)
    // do not use early return here
    if (gIdx < Threads)
#   endif
    {
        if(!get_local_id(1))
        {
            __local ulong* State = State_buf + get_local_id(0) * 25;

            for(int i = 0; i < 25; ++i) State[i] = states[i];

            keccakf1600_2(State);

            for(int i = 0; i < 25; ++i) states[i] = State[i];

            uint StateSwitch = State[0] & 3;
            __global uint *destinationBranch1 = StateSwitch == 0 ? Branch0 : Branch1;
            __global uint *destinationBranch2 = StateSwitch == 2 ? Branch2 : Branch3;
            __global uint *destinationBranch = StateSwitch < 2 ? destinationBranch1 : destinationBranch2;
            destinationBranch[atomic_inc(destinationBranch + Threads)] = gIdx;
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}

)==="
R"===(

#define VSWAP8(x)   (((x) >> 56) | (((x) >> 40) & 0x000000000000FF00UL) | (((x) >> 24) & 0x0000000000FF0000UL) \
          | (((x) >>  8) & 0x00000000FF000000UL) | (((x) <<  8) & 0x000000FF00000000UL) \
          | (((x) << 24) & 0x0000FF0000000000UL) | (((x) << 40) & 0x00FF000000000000UL) | (((x) << 56) & 0xFF00000000000000UL))

#define VSWAP4(x)   ((((x) >> 24) & 0xFFU) | (((x) >> 8) & 0xFF00U) | (((x) << 8) & 0xFF0000U) | (((x) << 24) & 0xFF000000U))

__kernel void Skein(__global ulong *states, __global uint *BranchBuf, __global uint *output, ulong Target, uint Threads)
{
    const uint idx = get_global_id(0) - get_global_offset(0);

    // do not use early return here
    if(idx < Threads)
    {
        states += 25 * BranchBuf[idx];

        // skein
        ulong8 h = vload8(0, SKEIN512_256_IV);

        // Type field begins with final bit, first bit, then six bits of type; the last 96
        // bits are input processed (including in the block to be processed with that tweak)
        // The output transform is only one run of UBI, since we need only 256 bits of output
        // The tweak for the output transform is Type = Output with the Final bit set
        // T[0] for the output is 8, and I don't know why - should be message size...
        ulong t[3] = { 0x00UL, 0x7000000000000000UL, 0x00UL };
        ulong8 p, m;

        #pragma unroll 1
        for (uint i = 0; i < 4; ++i)
        {
            t[0] += i < 3 ? 0x40UL : 0x08UL;

            t[2] = t[0] ^ t[1];

            m = (i < 3) ? vload8(i, states) : (ulong8)(states[24], 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL);
            const ulong h8 = h.s0 ^ h.s1 ^ h.s2 ^ h.s3 ^ h.s4 ^ h.s5 ^ h.s6 ^ h.s7 ^ SKEIN_KS_PARITY;
            p = Skein512Block(m, h, h8, t);

            h = m ^ p;

            t[1] = i < 2 ? 0x3000000000000000UL : 0xB000000000000000UL;
        }

        t[0] = 0x08UL;
        t[1] = 0xFF00000000000000UL;
        t[2] = t[0] ^ t[1];

        p = (ulong8)(0);
        const ulong h8 = h.s0 ^ h.s1 ^ h.s2 ^ h.s3 ^ h.s4 ^ h.s5 ^ h.s6 ^ h.s7 ^ SKEIN_KS_PARITY;

        p = Skein512Block(p, h, h8, t);

        // Note that comparison is equivalent to subtraction - we can't just compare 8 32-bit values
        // and expect an accurate result for target > 32-bit without implementing carries
        if (p.s3 <= Target) {
            ulong outIdx = atomic_inc(output + 0xFF);
            if (outIdx < 0xFF) {
                output[outIdx] = BranchBuf[idx] + (uint) get_global_offset(0);
            }
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}

#define SWAP8(x)    as_ulong(as_uchar8(x).s76543210)

#define JHXOR \
    h0h ^= input[0]; \
    h0l ^= input[1]; \
    h1h ^= input[2]; \
    h1l ^= input[3]; \
    h2h ^= input[4]; \
    h2l ^= input[5]; \
    h3h ^= input[6]; \
    h3l ^= input[7]; \
\
    E8; \
\
    h4h ^= input[0]; \
    h4l ^= input[1]; \
    h5h ^= input[2]; \
    h5l ^= input[3]; \
    h6h ^= input[4]; \
    h6l ^= input[5]; \
    h7h ^= input[6]; \
    h7l ^= input[7]

__kernel void JH(__global ulong *states, __global uint *BranchBuf, __global uint *output, ulong Target, uint Threads)
{
    const uint idx = get_global_id(0) - get_global_offset(0);

    // do not use early return here
    if(idx < Threads)
    {
        states += 25 * BranchBuf[idx];

        sph_u64 h0h = 0xEBD3202C41A398EBUL, h0l = 0xC145B29C7BBECD92UL, h1h = 0xFAC7D4609151931CUL, h1l = 0x038A507ED6820026UL, h2h = 0x45B92677269E23A4UL, h2l = 0x77941AD4481AFBE0UL, h3h = 0x7A176B0226ABB5CDUL, h3l = 0xA82FFF0F4224F056UL;
        sph_u64 h4h = 0x754D2E7F8996A371UL, h4l = 0x62E27DF70849141DUL, h5h = 0x948F2476F7957627UL, h5l = 0x6C29804757B6D587UL, h6h = 0x6C0D8EAC2D275E5CUL, h6l = 0x0F7A0557C6508451UL, h7h = 0xEA12247067D3E47BUL, h7l = 0x69D71CD313ABE389UL;
        sph_u64 tmp;

        for (uint i = 0; i < 3; ++i) {
            ulong input[8];

            const int shifted = i << 3;
            for (uint x = 0; x < 8; ++x) {
                input[x] = (states[shifted + x]);
            }

            JHXOR;
        }

        {
            ulong input[8] = { (states[24]), 0x80UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL };
            JHXOR;
        }

        {
            ulong input[8] = { 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x00UL, 0x4006000000000000UL };
            JHXOR;
        }

        // Note that comparison is equivalent to subtraction - we can't just compare 8 32-bit values
        // and expect an accurate result for target > 32-bit without implementing carries
        if (h7l <= Target) {
            ulong outIdx = atomic_inc(output + 0xFF);
            if (outIdx < 0xFF) {
                output[outIdx] = BranchBuf[idx] + (uint) get_global_offset(0);
            }
        }
    }
}

#define SWAP4(x)    as_uint(as_uchar4(x).s3210)

__kernel void Blake(__global ulong *states, __global uint *BranchBuf, __global uint *output, ulong Target, uint Threads)
{
    const uint idx = get_global_id(0) - get_global_offset(0);

    // do not use early return here
    if (idx < Threads)
    {
        states += 25 * BranchBuf[idx];

        unsigned int m[16];
        unsigned int v[16];
        uint h[8];
        uint bitlen = 0;

        ((uint8 *)h)[0] = vload8(0U, c_IV256);

        for (uint i = 0; i < 3; ++i) {
            ((uint16 *)m)[0] = vload16(i, (__global uint *)states);
            for (uint x = 0; x < 16; ++x) {
                m[x] = SWAP4(m[x]);
            }

            bitlen += 512;

            ((uint16 *)v)[0].lo = ((uint8 *)h)[0];
            ((uint16 *)v)[0].hi = vload8(0U, c_u256);

            v[12] ^= bitlen;
            v[13] ^= bitlen;

            for (uint r = 0; r < 14; r++) {
                GS(0, 4, 0x8, 0xC, 0x0);
                GS(1, 5, 0x9, 0xD, 0x2);
                GS(2, 6, 0xA, 0xE, 0x4);
                GS(3, 7, 0xB, 0xF, 0x6);
                GS(0, 5, 0xA, 0xF, 0x8);
                GS(1, 6, 0xB, 0xC, 0xA);
                GS(2, 7, 0x8, 0xD, 0xC);
                GS(3, 4, 0x9, 0xE, 0xE);
            }

            ((uint8 *)h)[0] ^= ((uint8 *)v)[0] ^ ((uint8 *)v)[1];
        }

        m[0]  = SWAP4(((__global uint *)states)[48]);
        m[1]  = SWAP4(((__global uint *)states)[49]);
        m[2]  = 0x80000000U;
        m[3]  = 0x00U;
        m[4]  = 0x00U;
        m[5]  = 0x00U;
        m[6]  = 0x00U;
        m[7]  = 0x00U;
        m[8]  = 0x00U;
        m[9]  = 0x00U;
        m[10] = 0x00U;
        m[11] = 0x00U;
        m[12] = 0x00U;
        m[13] = 1U;
        m[14] = 0U;
        m[15] = 0x640;

        bitlen += 64;

        ((uint16 *)v)[0].lo = ((uint8 *)h)[0];
        ((uint16 *)v)[0].hi = vload8(0U, c_u256);

        v[12] ^= bitlen;
        v[13] ^= bitlen;

        for (uint r = 0; r < 14; r++) {
            GS(0, 4, 0x8, 0xC, 0x0);
            GS(1, 5, 0x9, 0xD, 0x2);
            GS(2, 6, 0xA, 0xE, 0x4);
            GS(3, 7, 0xB, 0xF, 0x6);
            GS(0, 5, 0xA, 0xF, 0x8);
            GS(1, 6, 0xB, 0xC, 0xA);
            GS(2, 7, 0x8, 0xD, 0xC);
            GS(3, 4, 0x9, 0xE, 0xE);
        }

        ((uint8 *)h)[0] ^= ((uint8 *)v)[0] ^ ((uint8 *)v)[1];

        for (uint i = 0; i < 8; ++i) {
            h[i] = SWAP4(h[i]);
        }

        // Note that comparison is equivalent to subtraction - we can't just compare 8 32-bit values
        // and expect an accurate result for target > 32-bit without implementing carries
        uint2 t = (uint2)(h[6],h[7]);
        if (as_ulong(t) <= Target) {
            ulong outIdx = atomic_inc(output + 0xFF);
            if (outIdx < 0xFF) {
                output[outIdx] = BranchBuf[idx] + (uint) get_global_offset(0);
            }
        }
    }
}

#undef SWAP4


__kernel void Groestl(__global ulong *states, __global uint *BranchBuf, __global uint *output, ulong Target, uint Threads)
{
    const uint idx = get_global_id(0) - get_global_offset(0);

    // do not use early return here
    if (idx < Threads)
    {
        states += 25 * BranchBuf[idx];

        ulong State[8] = { 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0x0001000000000000UL };
        ulong H[8], M[8];

        // BUG: AMD driver 19.7.X crashs if this is written as loop
        // Thx AMD for so bad software
        {
            ((ulong8 *)M)[0] = vload8(0, states);

            for (uint x = 0; x < 8; ++x) {
                H[x] = M[x] ^ State[x];
            }

            PERM_SMALL_P(H);
            PERM_SMALL_Q(M);

            for (uint x = 0; x < 8; ++x) {
                State[x] ^= H[x] ^ M[x];
            }
        }

        {
            ((ulong8 *)M)[0] = vload8(1, states);

            for (uint x = 0; x < 8; ++x) {
                H[x] = M[x] ^ State[x];
            }

            PERM_SMALL_P(H);
            PERM_SMALL_Q(M);

            for (uint x = 0; x < 8; ++x) {
                State[x] ^= H[x] ^ M[x];
            }
        }

        {
            ((ulong8 *)M)[0] = vload8(2, states);

            for (uint x = 0; x < 8; ++x) {
                H[x] = M[x] ^ State[x];
            }

            PERM_SMALL_P(H);
            PERM_SMALL_Q(M);

            for (uint x = 0; x < 8; ++x) {
                State[x] ^= H[x] ^ M[x];
            }
        }

        M[0] = states[24];
        M[1] = 0x80UL;
        M[2] = 0UL;
        M[3] = 0UL;
        M[4] = 0UL;
        M[5] = 0UL;
        M[6] = 0UL;
        M[7] = 0x0400000000000000UL;

        for (uint x = 0; x < 8; ++x) {
            H[x] = M[x] ^ State[x];
        }

        PERM_SMALL_P(H);
        PERM_SMALL_Q(M);

        ulong tmp[8];
        for (uint i = 0; i < 8; ++i) {
            tmp[i] = State[i] ^= H[i] ^ M[i];
        }

        PERM_SMALL_P(State);

        for (uint i = 0; i < 8; ++i) {
            State[i] ^= tmp[i];
        }

        // Note that comparison is equivalent to subtraction - we can't just compare 8 32-bit values
        // and expect an accurate result for target > 32-bit without implementing carries
        if (State[7] <= Target) {
            ulong outIdx = atomic_inc(output + 0xFF);
            if (outIdx < 0xFF) {
                output[outIdx] = BranchBuf[idx] + (uint) get_global_offset(0);
            }
        }
    }
}

)==="
