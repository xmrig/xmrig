#ifndef XMRIG_KECCAK_CL
#define XMRIG_KECCAK_CL


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
    for (round = 0; round < 24; ++round) {
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
        #pragma unroll 1
        for (i = 0; i < 24; ++i) {
            bc[0] = st[keccakf_piln[i]];
            st[keccakf_piln[i]] = rotate(t, (ulong)keccakf_rotc[i]);
            t = bc[0];
        }

        #pragma unroll 1
        for (int i = 0; i < 25; i += 5) {
            ulong tmp[5];

            #pragma unroll 1
            for (int x = 0; x < 5; ++x) {
                tmp[x] = bitselect(st[i + x] ^ st[i + ((x + 2) % 5)], st[i + x], st[i + ((x + 1) % 5)]);
            }

            #pragma unroll 1
            for (int x = 0; x < 5; ++x) {
                st[i + x] = tmp[x];
            }
        }

        //  Iota
        st[0] ^= keccakf_rndc[round];
    }
}


void keccakf1600_2(__local ulong *st)
{
    int i, round;
    ulong t, bc[5];

    #pragma unroll 1
    for (round = 0; round < 24; ++round) {
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20] ^ rotate(st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22], 1UL);
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21] ^ rotate(st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23], 1UL);
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22] ^ rotate(st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24], 1UL);
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23] ^ rotate(st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20], 1UL);
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24] ^ rotate(st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21], 1UL);

        st[0]  ^= bc[4];
        st[5]  ^= bc[4];
        st[10] ^= bc[4];
        st[15] ^= bc[4];
        st[20] ^= bc[4];

        st[1]  ^= bc[0];
        st[6]  ^= bc[0];
        st[11] ^= bc[0];
        st[16] ^= bc[0];
        st[21] ^= bc[0];

        st[2]  ^= bc[1];
        st[7]  ^= bc[1];
        st[12] ^= bc[1];
        st[17] ^= bc[1];
        st[22] ^= bc[1];

        st[3]  ^= bc[2];
        st[8]  ^= bc[2];
        st[13] ^= bc[2];
        st[18] ^= bc[2];
        st[23] ^= bc[2];

        st[4]  ^= bc[3];
        st[9]  ^= bc[3];
        st[14] ^= bc[3];
        st[19] ^= bc[3];
        st[24] ^= bc[3];

        // Rho Pi
        t = st[1];
        #pragma unroll 1
        for (i = 0; i < 24; ++i) {
            bc[0] = st[keccakf_piln[i]];
            st[keccakf_piln[i]] = rotate(t, (ulong)keccakf_rotc[i]);
            t = bc[0];
        }

        #pragma unroll 1
        for (int i = 0; i < 25; i += 5) {
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


#endif
