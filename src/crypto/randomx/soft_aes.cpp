/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>
Copyright (c) 2019 SChernykh   <https://github.com/SChernykh>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "crypto/randomx/soft_aes.h"

alignas(64) uint32_t lutEnc0[256];
alignas(64) uint32_t lutEnc1[256];
alignas(64) uint32_t lutEnc2[256];
alignas(64) uint32_t lutEnc3[256];

alignas(64) uint32_t lutDec0[256];
alignas(64) uint32_t lutDec1[256];
alignas(64) uint32_t lutDec2[256];
alignas(64) uint32_t lutDec3[256];

alignas(64) uint8_t lutEncIndex[4][32];
alignas(64) uint8_t lutDecIndex[4][32];

static uint32_t mul_gf2(uint32_t b, uint32_t c)
{
	uint32_t s = 0;
	for (uint32_t i = b, j = c, k = 1; (k < 0x100) && j; k <<= 1)
	{
		if (j & k)
		{
			s ^= i;
			j ^= k;
		}

		i <<= 1;
		if (i & 0x100)
			i ^= (1 << 8) | (1 << 4) | (1 << 3) | (1 << 1) | (1 << 0);
	}

	return s;
}

#define ROTL8(x,shift) ((uint8_t) ((x) << (shift)) | ((x) >> (8 - (shift))))

static struct SAESInitializer
{
	SAESInitializer()
	{
		static uint8_t sbox[256];
		static uint8_t sbox_reverse[256];

		uint8_t p = 1, q = 1;

		do {
			p = p ^ (p << 1) ^ (p & 0x80 ? 0x1B : 0);

			q ^= q << 1;
			q ^= q << 2;
			q ^= q << 4;
			q ^= (q & 0x80) ? 0x09 : 0;

			const uint8_t value = q ^ ROTL8(q, 1) ^ ROTL8(q, 2) ^ ROTL8(q, 3) ^ ROTL8(q, 4) ^ 0x63;
			sbox[p] = value;
			sbox_reverse[value] = p;
		} while (p != 1);

		sbox[0] = 0x63;
		sbox_reverse[0x63] = 0;

		for (uint32_t i = 0; i < 0x100; ++i)
		{
			union
			{
				uint32_t w;
				uint8_t p[4];
			};

			uint32_t s = sbox[i];
			p[0] = mul_gf2(s, 2);
			p[1] = s;
			p[2] = s;
			p[3] = mul_gf2(s, 3);

			lutEnc0[i] = w; w = (w << 8) | (w >> 24);
			lutEnc1[i] = w; w = (w << 8) | (w >> 24);
			lutEnc2[i] = w; w = (w << 8) | (w >> 24);
			lutEnc3[i] = w;

			s = sbox_reverse[i];
			p[0] = mul_gf2(s, 0xe);
			p[1] = mul_gf2(s, 0x9);
			p[2] = mul_gf2(s, 0xd);
			p[3] = mul_gf2(s, 0xb);

			lutDec0[i] = w; w = (w << 8) | (w >> 24);
			lutDec1[i] = w; w = (w << 8) | (w >> 24);
			lutDec2[i] = w; w = (w << 8) | (w >> 24);
			lutDec3[i] = w;
		}

		memset(lutEncIndex, -1, sizeof(lutEncIndex));
		memset(lutDecIndex, -1, sizeof(lutDecIndex));

		lutEncIndex[0][ 0] =  0;
		lutEncIndex[0][ 4] =  4;
		lutEncIndex[0][ 8] =  8;
		lutEncIndex[0][12] = 12;
		lutEncIndex[1][ 0] =  5;
		lutEncIndex[1][ 4] =  9;
		lutEncIndex[1][ 8] = 13;
		lutEncIndex[1][12] =  1;
		lutEncIndex[2][ 0] = 10;
		lutEncIndex[2][ 4] = 14;
		lutEncIndex[2][ 8] =  2;
		lutEncIndex[2][12] =  6;
		lutEncIndex[3][ 0] = 15;
		lutEncIndex[3][ 4] =  3;
		lutEncIndex[3][ 8] =  7;
		lutEncIndex[3][12] = 11;

		lutDecIndex[0][ 0] =  0;
		lutDecIndex[0][ 4] =  4;
		lutDecIndex[0][ 8] =  8;
		lutDecIndex[0][12] = 12;
		lutDecIndex[1][ 0] = 13;
		lutDecIndex[1][ 4] =  1;
		lutDecIndex[1][ 8] =  5;
		lutDecIndex[1][12] =  9;
		lutDecIndex[2][ 0] = 10;
		lutDecIndex[2][ 4] = 14;
		lutDecIndex[2][ 8] =  2;
		lutDecIndex[2][12] =  6;
		lutDecIndex[3][ 0] =  7;
		lutDecIndex[3][ 4] = 11;
		lutDecIndex[3][ 8] = 15;
		lutDecIndex[3][12] =  3;

		for (uint32_t i = 0; i < 4; ++i) {
			for (uint32_t j = 0; j < 16; j += 4) {
				lutEncIndex[i][j + 16] = lutEncIndex[i][j] + 16;
				lutDecIndex[i][j + 16] = lutDecIndex[i][j] + 16;
			}
		}
	}
} aes_initializer;
