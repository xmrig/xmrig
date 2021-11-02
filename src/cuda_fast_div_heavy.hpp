#pragma once

#include <stdint.h>

__device__ __forceinline__ uint64_t fast_div_heavy(int64_t _a, int32_t _b)
{
	int64_t a = abs(_a);
	int32_t b = abs(_b);

	float rcp;
	asm("rcp.approx.f32 %0, %1;" : "=f"(rcp) : "f"(__int2float_rn(b)));
	float rcp2 = __uint_as_float(__float_as_uint(rcp) + (32U << 23));

	uint64_t q1 = __float2ull_rd(__int2float_rn(((int32_t*)&a)[1]) * rcp2);
	a -= q1 * (uint32_t)(b);

	rcp2 = __uint_as_float(__float_as_uint(rcp) + (12U << 23));
	int64_t q2 = __float2ll_rn(__int2float_rn(a >> 12) * rcp2);
	int32_t a2 = ((int32_t*)&a)[0] - ((int32_t*)&q2)[0] * b;

	int32_t q3 = __float2int_rn(__int2float_rn(a2) * rcp);
	q3 += (a2 - q3 * b) >> 31;

	const int64_t q = q1 + q2 + q3;
	return ((((int32_t*)&_a)[1] ^ _b) < 0) ? -q : q;
}
