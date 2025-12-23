/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>

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

#include <assert.h>
#include "crypto/randomx/reciprocal.h"

/*
	Calculates rcp = 2**x / divisor for highest integer x such that rcp < 2**64.
	divisor must not be 0 or a power of 2

	Equivalent x86 assembly (divisor in rcx):

	mov edx, 1
	mov r8, rcx
	xor eax, eax
	bsr rcx, rcx
	shl rdx, cl
	div r8
	ret

*/
uint64_t randomx_reciprocal(uint64_t divisor) {

	assert(divisor != 0);

	const uint64_t p2exp63 = 1ULL << 63;

	uint64_t quotient = p2exp63 / divisor, remainder = p2exp63 % divisor;

	unsigned bsr = 0; //highest set bit in divisor

	for (uint64_t bit = divisor; bit > 0; bit >>= 1)
		bsr++;

	for (unsigned shift = 0; shift < bsr; shift++) {
		if (remainder >= divisor - remainder) {
			quotient = quotient * 2 + 1;
			remainder = remainder * 2 - divisor;
		}
		else {
			quotient = quotient * 2;
			remainder = remainder * 2;
		}
	}

	return quotient;
}

#if !RANDOMX_HAVE_FAST_RECIPROCAL

#ifdef __GNUC__
uint64_t randomx_reciprocal_fast(uint64_t divisor)
{
	const uint64_t q = (1ULL << 63) / divisor;
	const uint64_t r = (1ULL << 63) % divisor;

	const uint64_t shift = 64 - __builtin_clzll(divisor);

	return (q << shift) + ((r << shift) / divisor);
}
#else
uint64_t randomx_reciprocal_fast(uint64_t divisor) {
	return randomx_reciprocal(divisor);
}
#endif

#endif
