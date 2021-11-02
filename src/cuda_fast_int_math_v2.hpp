#pragma once

#include <stdint.h>

__device__ __forceinline__ uint32_t get_reciprocal(uint32_t a)
{
	const float a_hi = __uint_as_float((a >> 8) + ((126U + 31U) << 23));
	const float a_lo = __uint2float_rn(a & 0xFF);

	float r;
	asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(a_hi));
	const float r_scaled = __uint_as_float(__float_as_uint(r) + (64U << 23));

	const float h = __fmaf_rn(a_lo, r, __fmaf_rn(a_hi, r, -1.0f));
	return (__float_as_uint(r) << 9) - __float2int_rn(h * r_scaled);
}

__device__ __forceinline__ uint64_t fast_div_v2(uint64_t a, uint32_t b)
{
	const uint32_t r = get_reciprocal(b);
	const uint64_t k = __umulhi(((uint32_t*)&a)[0], r) + ((uint64_t)(r) * ((uint32_t*)&a)[1]) + a;

	uint32_t q[2];
	q[0] = ((uint32_t*)&k)[1];

	int64_t tmp = a - (uint64_t)(q[0]) * b;
	((int32_t*)(&tmp))[1] -= (k < a) ? b : 0;

	const bool overshoot = ((int32_t*)(&tmp))[1] < 0;
	const bool undershoot = tmp >= b;

	q[0] += (undershoot ? 1U : 0U) - (overshoot ? 1U : 0U);
	q[1] = ((uint32_t*)(&tmp))[0] + (overshoot ? b : 0U) - (undershoot ? b : 0U);

	return *((uint64_t*)(q));
}

__device__ __forceinline__ uint32_t fast_sqrt_v2(const uint64_t n1)
{
	float x = __uint_as_float((((uint32_t*)&n1)[1] >> 9) + ((64U + 127U) << 23));
	float x1;
	asm("rsqrt.approx.f32 %0, %1;" : "=f"(x1) : "f"(x));
	asm("sqrt.approx.f32 %0, %1;" : "=f"(x) : "f"(x));

	// The following line does x1 *= 4294967296.0f;
	x1 = __uint_as_float(__float_as_uint(x1) + (32U << 23));

	const uint32_t x0 = __float_as_uint(x) - (158U << 23);
	const int64_t delta0 = n1 - (((int64_t)(x0) * x0) << 18);
	const float delta = __int2float_rn(((int32_t*)&delta0)[1]) * x1;

	uint32_t result = (x0 << 10) + __float2int_rn(delta);
	const uint32_t s = result >> 1;
	const uint32_t b = result & 1;

	const uint64_t x2 = (uint64_t)(s) * (s + b) + ((uint64_t)(result) << 32) - n1;
	if ((int64_t)(x2 + b) > 0) --result;
	if ((int64_t)(x2 + 0x100000000UL + s) < 0) ++result;

	return result;
}
