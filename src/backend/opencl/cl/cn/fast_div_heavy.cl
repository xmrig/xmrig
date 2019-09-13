#ifndef FAST_DIV_HEAVY_CL
#define FAST_DIV_HEAVY_CL

#if (ALGO_FAMILY == FAMILY_CN_HEAVY)

inline long fast_div_heavy(long _a, int _b)
{
	long a = abs(_a);
	int b = abs(_b);

	float rcp = native_recip(convert_float_rte(b));
	float rcp2 = as_float(as_uint(rcp) + (32U << 23));

	ulong q1 = convert_ulong(convert_float_rte(as_int2(a).s1) * rcp2);
	a -= q1 * as_uint(b);

	float q2f = convert_float_rte(as_int2(a >> 12).s0) * rcp;
	q2f = as_float(as_uint(q2f) + (12U << 23));
	long q2 = convert_long_rte(q2f);
	int a2 = as_int2(a).s0 - as_int2(q2).s0 * b;

	int q3 = convert_int_rte(convert_float_rte(a2) * rcp);
	q3 += (a2 - q3 * b) >> 31;

	const long q = q1 + q2 + q3;
	return ((as_int2(_a).s1 ^ _b) < 0) ? -q : q;
}

#endif

#endif
