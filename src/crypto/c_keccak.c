// keccak.c
// 19-Nov-11  Markku-Juhani O. Saarinen <mjos@iki.fi>
// A baseline Keccak (3rd round) implementation.

#include <stdint.h>
#include <memory.h>

#define HASH_DATA_AREA 136
#define KECCAK_ROUNDS 24

#ifndef ROTL64
#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))
#endif

const uint64_t keccakf_rndc[24] = 
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

// update the state with given number of rounds

void keccakf(uint64_t st[25], int rounds)
{
    int i, j, round;
    uint64_t t, bc[5];

    for (round = 0; round < rounds; ++round) {

        // Theta
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        for (i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            st[i     ] ^= t;
            st[i +  5] ^= t;
            st[i + 10] ^= t;
            st[i + 15] ^= t;
            st[i + 20] ^= t;
        }

        // Rho Pi
        t = st[1];
        st[ 1] = ROTL64(st[ 6], 44);
        st[ 6] = ROTL64(st[ 9], 20);
        st[ 9] = ROTL64(st[22], 61);
        st[22] = ROTL64(st[14], 39);
        st[14] = ROTL64(st[20], 18);
        st[20] = ROTL64(st[ 2], 62);
        st[ 2] = ROTL64(st[12], 43);
        st[12] = ROTL64(st[13], 25);
        st[13] = ROTL64(st[19],  8);
        st[19] = ROTL64(st[23], 56);
        st[23] = ROTL64(st[15], 41);
        st[15] = ROTL64(st[ 4], 27);
        st[ 4] = ROTL64(st[24], 14);
        st[24] = ROTL64(st[21],  2);
        st[21] = ROTL64(st[ 8], 55);
        st[ 8] = ROTL64(st[16], 45);
        st[16] = ROTL64(st[ 5], 36);
        st[ 5] = ROTL64(st[ 3], 28);
        st[ 3] = ROTL64(st[18], 21);
        st[18] = ROTL64(st[17], 15);
        st[17] = ROTL64(st[11], 10);
        st[11] = ROTL64(st[ 7],  6);
        st[ 7] = ROTL64(st[10],  3);
        st[10] = ROTL64(t, 1);

        //  Chi
        // unrolled loop, where only last iteration is different
        j = 0;
        bc[0] = st[j    ];
        bc[1] = st[j + 1];

        st[j    ] ^= (~st[j + 1]) & st[j + 2];
        st[j + 1] ^= (~st[j + 2]) & st[j + 3];
        st[j + 2] ^= (~st[j + 3]) & st[j + 4];
        st[j + 3] ^= (~st[j + 4]) & bc[0];
        st[j + 4] ^= (~bc[0]) & bc[1];

        j = 5;
        bc[0] = st[j    ];
        bc[1] = st[j + 1];

        st[j    ] ^= (~st[j + 1]) & st[j + 2];
        st[j + 1] ^= (~st[j + 2]) & st[j + 3];
        st[j + 2] ^= (~st[j + 3]) & st[j + 4];
        st[j + 3] ^= (~st[j + 4]) & bc[0];
        st[j + 4] ^= (~bc[0]) & bc[1];

        j = 10;
        bc[0] = st[j    ];
        bc[1] = st[j + 1];

        st[j    ] ^= (~st[j + 1]) & st[j + 2];
        st[j + 1] ^= (~st[j + 2]) & st[j + 3];
        st[j + 2] ^= (~st[j + 3]) & st[j + 4];
        st[j + 3] ^= (~st[j + 4]) & bc[0];
        st[j + 4] ^= (~bc[0]) & bc[1];

        j = 15;
        bc[0] = st[j    ];
        bc[1] = st[j + 1];

        st[j    ] ^= (~st[j + 1]) & st[j + 2];
        st[j + 1] ^= (~st[j + 2]) & st[j + 3];
        st[j + 2] ^= (~st[j + 3]) & st[j + 4];
        st[j + 3] ^= (~st[j + 4]) & bc[0];
        st[j + 4] ^= (~bc[0]) & bc[1];

        j = 20;
        bc[0] = st[j    ];
        bc[1] = st[j + 1];
        bc[2] = st[j + 2];
        bc[3] = st[j + 3];
        bc[4] = st[j + 4];

        st[j    ] ^= (~bc[1]) & bc[2];
        st[j + 1] ^= (~bc[2]) & bc[3];
        st[j + 2] ^= (~bc[3]) & bc[4];
        st[j + 3] ^= (~bc[4]) & bc[0];
        st[j + 4] ^= (~bc[0]) & bc[1];
        
        //  Iota
        st[0] ^= keccakf_rndc[round];
    }
}

// compute a keccak hash (md) of given byte length from "in"
typedef uint64_t state_t[25];

void keccak(const uint8_t *in, int inlen, uint8_t *md, int mdlen)
{
    state_t st;
    uint8_t temp[144];
    int i, rsiz, rsizw;

    rsiz = sizeof(state_t) == mdlen ? HASH_DATA_AREA : 200 - 2 * mdlen;
    rsizw = rsiz / 8;
    
    memset(st, 0, sizeof(st));

    for ( ; inlen >= rsiz; inlen -= rsiz, in += rsiz) {
        for (i = 0; i < rsizw; i++)
            st[i] ^= ((uint64_t *) in)[i];
        keccakf(st, KECCAK_ROUNDS);
    }
    
    // last block and padding
    memcpy(temp, in, inlen);
    temp[inlen++] = 1;
    memset(temp + inlen, 0, rsiz - inlen);
    temp[rsiz - 1] |= 0x80;

    for (i = 0; i < rsizw; i++)
        st[i] ^= ((uint64_t *) temp)[i];

    keccakf(st, KECCAK_ROUNDS);

    memcpy(md, st, mdlen);
}

void keccak1600(const uint8_t *in, int inlen, uint8_t *md)
{
    keccak(in, inlen, md, sizeof(state_t));
}
