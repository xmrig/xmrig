/*
 * Based on public domain code available at: http://cr.yp.to/snuffle.html
 *
 * Modifications and C-native SSE macro based SSE implementation by
 * Adam Ierymenko <adam.ierymenko@zerotier.com>.
 *
 * Additional modifications and code cleanup for AstroBWT by
 * SChernykh <https://github.com/SChernykh>
 *
 * Since the original was public domain, this is too.
 */

#include "Salsa20.hpp"

// Statically compute and define SSE constants
class _s20sseconsts
{
public:
	_s20sseconsts()
	{
		maskLo32 = _mm_shuffle_epi32(_mm_cvtsi32_si128(-1), _MM_SHUFFLE(1, 0, 1, 0));
		maskHi32 = _mm_slli_epi64(maskLo32, 32);
	}
	__m128i maskLo32,maskHi32;
};
static const _s20sseconsts _S20SSECONSTANTS;

namespace ZeroTier {

void Salsa20::init(const void *key,const void *iv)
{
	const uint32_t *const k = (const uint32_t *)key;
	_state.i[0] = 0x61707865;
	_state.i[1] = 0x3320646e;
	_state.i[2] = 0x79622d32;
	_state.i[3] = 0x6b206574;
	_state.i[4] = k[3];
	_state.i[5] = 0;
	_state.i[6] = k[7];
	_state.i[7] = k[2];
	_state.i[8] = 0;
	_state.i[9] = k[6];
	_state.i[10] = k[1];
	_state.i[11] = ((const uint32_t *)iv)[1];
	_state.i[12] = k[5];
	_state.i[13] = k[0];
	_state.i[14] = ((const uint32_t *)iv)[0];
	_state.i[15] = k[4];
}

void Salsa20::XORKeyStream(void *out,unsigned int bytes)
{
	uint8_t tmp[64];
	uint8_t *c = (uint8_t *)out;
	uint8_t *ctarget = c;
	unsigned int i;

	if (!bytes)
		return;

	for (;;) {
		if (bytes < 64) {
			for (i = 0;i < bytes;++i)
				tmp[i] = 0;
			ctarget = c;
			c = tmp;
		}

		__m128i X0 = _mm_loadu_si128((const __m128i *)&(_state.v[0]));
		__m128i X1 = _mm_loadu_si128((const __m128i *)&(_state.v[1]));
		__m128i X2 = _mm_loadu_si128((const __m128i *)&(_state.v[2]));
		__m128i X3 = _mm_loadu_si128((const __m128i *)&(_state.v[3]));
		__m128i T;
		__m128i X0s = X0;
		__m128i X1s = X1;
		__m128i X2s = X2;
		__m128i X3s = X3;

		// 2X round -------------------------------------------------------------
		T = _mm_add_epi32(X0, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X1, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X3, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x93);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x39);
		T = _mm_add_epi32(X0, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X3, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X1, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x39);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x93);

		// 2X round -------------------------------------------------------------
		T = _mm_add_epi32(X0, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X1, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X3, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x93);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x39);
		T = _mm_add_epi32(X0, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X3, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X1, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x39);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x93);

		// 2X round -------------------------------------------------------------
		T = _mm_add_epi32(X0, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X1, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X3, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x93);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x39);
		T = _mm_add_epi32(X0, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X3, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X1, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x39);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x93);

		// 2X round -------------------------------------------------------------
		T = _mm_add_epi32(X0, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X1, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X3, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x93);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x39);
		T = _mm_add_epi32(X0, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X3, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X1, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x39);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x93);

		// 2X round -------------------------------------------------------------
		T = _mm_add_epi32(X0, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X1, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X3, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x93);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x39);
		T = _mm_add_epi32(X0, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X3, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X1, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x39);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x93);

		// 2X round -------------------------------------------------------------
		T = _mm_add_epi32(X0, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X1, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X3, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x93);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x39);
		T = _mm_add_epi32(X0, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X3, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X1, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x39);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x93);

		// 2X round -------------------------------------------------------------
		T = _mm_add_epi32(X0, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X1, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X3, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x93);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x39);
		T = _mm_add_epi32(X0, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X3, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X1, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x39);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x93);

		// 2X round -------------------------------------------------------------
		T = _mm_add_epi32(X0, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X1, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X3, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x93);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x39);
		T = _mm_add_epi32(X0, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X3, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X1, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x39);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x93);

		// 2X round -------------------------------------------------------------
		T = _mm_add_epi32(X0, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X1, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X3, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x93);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x39);
		T = _mm_add_epi32(X0, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X3, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X1, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x39);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x93);

		// 2X round -------------------------------------------------------------
		T = _mm_add_epi32(X0, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X1, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X3, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x93);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x39);
		T = _mm_add_epi32(X0, X1);
		X3 = _mm_xor_si128(_mm_xor_si128(X3, _mm_slli_epi32(T, 7)), _mm_srli_epi32(T, 25));
		T = _mm_add_epi32(X3, X0);
		X2 = _mm_xor_si128(_mm_xor_si128(X2, _mm_slli_epi32(T, 9)), _mm_srli_epi32(T, 23));
		T = _mm_add_epi32(X2, X3);
		X1 = _mm_xor_si128(_mm_xor_si128(X1, _mm_slli_epi32(T, 13)), _mm_srli_epi32(T, 19));
		T = _mm_add_epi32(X1, X2);
		X0 = _mm_xor_si128(_mm_xor_si128(X0, _mm_slli_epi32(T, 18)), _mm_srli_epi32(T, 14));
		X1 = _mm_shuffle_epi32(X1, 0x39);
		X2 = _mm_shuffle_epi32(X2, 0x4E);
		X3 = _mm_shuffle_epi32(X3, 0x93);

		X0 = _mm_add_epi32(X0s,X0);
		X1 = _mm_add_epi32(X1s,X1);
		X2 = _mm_add_epi32(X2s,X2);
		X3 = _mm_add_epi32(X3s,X3);

		__m128i k02 = _mm_shuffle_epi32(_mm_or_si128(_mm_slli_epi64(X0, 32), _mm_srli_epi64(X3, 32)), _MM_SHUFFLE(0, 1, 2, 3));
		__m128i k13 = _mm_shuffle_epi32(_mm_or_si128(_mm_slli_epi64(X1, 32), _mm_srli_epi64(X0, 32)), _MM_SHUFFLE(0, 1, 2, 3));
		__m128i k20 = _mm_or_si128(_mm_and_si128(X2, _S20SSECONSTANTS.maskLo32), _mm_and_si128(X1, _S20SSECONSTANTS.maskHi32));
		__m128i k31 = _mm_or_si128(_mm_and_si128(X3, _S20SSECONSTANTS.maskLo32), _mm_and_si128(X2, _S20SSECONSTANTS.maskHi32));
		_mm_storeu_ps(reinterpret_cast<float *>(c),_mm_castsi128_ps(_mm_unpackhi_epi64(k02,k20)));
		_mm_storeu_ps(reinterpret_cast<float *>(c) + 4,_mm_castsi128_ps(_mm_unpackhi_epi64(k13,k31)));
		_mm_storeu_ps(reinterpret_cast<float *>(c) + 8,_mm_castsi128_ps(_mm_unpacklo_epi64(k20,k02)));
		_mm_storeu_ps(reinterpret_cast<float *>(c) + 12,_mm_castsi128_ps(_mm_unpacklo_epi64(k31,k13)));

		if (!(++_state.i[8])) {
			++_state.i[5]; // state reordered for SSE
			/* stopping at 2^70 bytes per nonce is user's responsibility */
		}

		if (bytes <= 64) {
			if (bytes < 64) {
				for (i = 0;i < bytes;++i)
					ctarget[i] = c[i];
			}

			return;
		}

		bytes -= 64;
		c += 64;
	}
}

} // namespace ZeroTier
