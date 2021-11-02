#pragma once

typedef struct {
	uint32_t h[8], s[4], t[2];
	uint32_t buflen;
	int nullt;
	uint8_t buf[64];
} blake_state;

#define U8TO32(p) \
	(((uint32_t)((p)[0]) << 24) | ((uint32_t)((p)[1]) << 16) | \
	((uint32_t)((p)[2]) <<  8) | ((uint32_t)((p)[3])      ))

#define U32TO8(p, v) \
	(p)[0] = (uint8_t)((v) >> 24); (p)[1] = (uint8_t)((v) >> 16); \
	(p)[2] = (uint8_t)((v) >>  8); (p)[3] = (uint8_t)((v)      );

#define BLAKE_ROT(x,n) ROTR32(x, n)
#define BLAKE_G(a,b,c,d,e) \
	v[a] += (m[d_blake_sigma[i][e]] ^ d_blake_cst[d_blake_sigma[i][e+1]]) + v[b]; \
	v[d] = BLAKE_ROT(v[d] ^ v[a],16); \
	v[c] += v[d];                     \
	v[b] = BLAKE_ROT(v[b] ^ v[c],12); \
	v[a] += (m[d_blake_sigma[i][e+1]] ^ d_blake_cst[d_blake_sigma[i][e]])+v[b]; \
	v[d] = BLAKE_ROT(v[d] ^ v[a], 8); \
	v[c] += v[d];                     \
	v[b] = BLAKE_ROT(v[b] ^ v[c], 7);

__constant__ uint8_t d_blake_sigma[14][16] =
{
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
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
	{7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8}
};
__constant__ uint32_t d_blake_cst[16]
= {
	0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
	0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
	0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
	0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
};

__device__ void cn_blake_compress(blake_state *S, const uint8_t *block)
{
	uint32_t v[16], m[16], i;

	for (i = 0; i < 16; ++i) m[i] = U8TO32(block + i * 4);
	for (i = 0; i < 8;  ++i) v[i] = S->h[i];
	v[ 8] = S->s[0] ^ 0x243F6A88;
	v[ 9] = S->s[1] ^ 0x85A308D3;
	v[10] = S->s[2] ^ 0x13198A2E;
	v[11] = S->s[3] ^ 0x03707344;
	v[12] = 0xA4093822;
	v[13] = 0x299F31D0;
	v[14] = 0x082EFA98;
	v[15] = 0xEC4E6C89;

	if (S->nullt == 0)
	{
		v[12] ^= S->t[0];
		v[13] ^= S->t[0];
		v[14] ^= S->t[1];
		v[15] ^= S->t[1];
	}

	for (i = 0; i < 14; ++i)
	{
		BLAKE_G(0, 4,  8, 12,  0);
		BLAKE_G(1, 5,  9, 13,  2);
		BLAKE_G(2, 6, 10, 14,  4);
		BLAKE_G(3, 7, 11, 15,  6);
		BLAKE_G(3, 4,  9, 14, 14);
		BLAKE_G(2, 7,  8, 13, 12);
		BLAKE_G(0, 5, 10, 15,  8);
		BLAKE_G(1, 6, 11, 12, 10);
	}

	for (i = 0; i < 16; ++i) S->h[i % 8] ^= v[i];
	for (i = 0; i < 8;  ++i) S->h[i] ^= S->s[i % 4];
}

__device__ void cn_blake_update(blake_state *S, const uint8_t *data, uint64_t datalen)
{
	uint32_t left = S->buflen >> 3;
	uint32_t fill = 64 - left;

	if (left && (((datalen >> 3) & 0x3F) >= fill))
	{
		memcpy((void *) (S->buf + left), (void *) data, fill);
		S->t[0] += 512;
		if (S->t[0] == 0) S->t[1]++;
		cn_blake_compress(S, S->buf);
		data += fill;
		datalen -= (fill << 3);
		left = 0;
	}

	while (datalen >= 512) 
	{
		S->t[0] += 512;
		if (S->t[0] == 0) S->t[1]++;
		cn_blake_compress(S, data);
		data += 64;
		datalen -= 512;
	}

	if (datalen > 0) 
	{
		memcpy((void *) (S->buf + left), (void *) data, datalen >> 3);
		S->buflen = (left << 3) + datalen;
	}
	else 
	{
		S->buflen = 0;
	}
}

__device__ void cn_blake_final(blake_state *S, uint8_t *digest)
{
	const uint8_t padding[] = 
	{
		0x80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
	};

	uint8_t pa = 0x81, pb = 0x01;
	uint8_t msglen[8];
	uint32_t lo = S->t[0] + S->buflen, hi = S->t[1];
	if (lo < (unsigned) S->buflen) hi++;
	U32TO8(msglen + 0, hi);
	U32TO8(msglen + 4, lo);

	if (S->buflen == 440) 
	{
		S->t[0] -= 8;
		cn_blake_update(S, &pa, 8);
	} 
	else 
	{
		if (S->buflen < 440) 
		{
			if (S->buflen == 0) S->nullt = 1;
			S->t[0] -= 440 - S->buflen;
			cn_blake_update(S, padding, 440 - S->buflen);
		}
		else 
		{
			S->t[0] -= 512 - S->buflen;
			cn_blake_update(S, padding, 512 - S->buflen);
			S->t[0] -= 440;
			cn_blake_update(S, padding + 1, 440);
			S->nullt = 1;
		}
		cn_blake_update(S, &pb, 8);
		S->t[0] -= 8;
	}
	S->t[0] -= 64;
	cn_blake_update(S, msglen, 64);

	U32TO8(digest +  0, S->h[0]);
	U32TO8(digest +  4, S->h[1]);
	U32TO8(digest +  8, S->h[2]);
	U32TO8(digest + 12, S->h[3]);
	U32TO8(digest + 16, S->h[4]);
	U32TO8(digest + 20, S->h[5]);
	U32TO8(digest + 24, S->h[6]);
	U32TO8(digest + 28, S->h[7]);
}

__device__ void cn_blake(const uint8_t *in, uint64_t inlen, uint8_t *out)
{
	blake_state bs;
	blake_state *S = (blake_state *)&bs;

	S->h[0] = 0x6A09E667; S->h[1] = 0xBB67AE85; S->h[2] = 0x3C6EF372;
	S->h[3] = 0xA54FF53A; S->h[4] = 0x510E527F; S->h[5] = 0x9B05688C;
	S->h[6] = 0x1F83D9AB; S->h[7] = 0x5BE0CD19;
	S->t[0] = S->t[1] = S->buflen = S->nullt = 0;
	S->s[0] = S->s[1] = S->s[2] = S->s[3] = 0;

	cn_blake_update(S, in, inlen * 8);
	cn_blake_final(S, out);
}
