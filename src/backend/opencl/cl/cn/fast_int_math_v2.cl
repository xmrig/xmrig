/*
 * @author SChernykh
 */

#if (ALGO_BASE == ALGO_CN_2)
inline uint get_reciprocal(uint a)
{
    const float a_hi = as_float((a >> 8) + ((126U + 31U) << 23));
    const float a_lo = convert_float_rte(a & 0xFF);

    const float r = native_recip(a_hi);
    const float r_scaled = as_float(as_uint(r) + (64U << 23));

    const float h = fma(a_lo, r, fma(a_hi, r, -1.0f));
    return (as_uint(r) << 9) - convert_int_rte(h * r_scaled);
}

inline uint2 fast_div_v2(ulong a, uint b)
{
    const uint r = get_reciprocal(b);
    const ulong k = mul_hi(as_uint2(a).s0, r) + ((ulong)(r) * as_uint2(a).s1) + a;

    const uint q = as_uint2(k).s1;
    long tmp = a - ((ulong)(q) * b);
    ((int*)&tmp)[1] -= (as_uint2(k).s1 < as_uint2(a).s1) ? b : 0;

    const int overshoot = ((int*)&tmp)[1] >> 31;
    const int undershoot = as_int2(as_uint(b - 1) - tmp).s1 >> 31;
    return (uint2)(q + overshoot - undershoot, as_uint2(tmp).s0 + (as_uint(overshoot) & b) - (as_uint(undershoot) & b));
}

inline uint fast_sqrt_v2(const ulong n1)
{
    float x = as_float((as_uint2(n1).s1 >> 9) + ((64U + 127U) << 23));

    float x1 = native_rsqrt(x);
    x = native_sqrt(x);

    // The following line does x1 *= 4294967296.0f;
    x1 = as_float(as_uint(x1) + (32U << 23));

    const uint x0 = as_uint(x) - (158U << 23);
    const long delta0 = n1 - (as_ulong((uint2)(mul24(x0, x0), mul_hi(x0, x0))) << 18);
    const float delta = convert_float_rte(as_int2(delta0).s1) * x1;

    uint result = (x0 << 10) + convert_int_rte(delta);
    const uint s = result >> 1;
    const uint b = result & 1;

    const ulong x2 = (ulong)(s) * (s + b) + ((ulong)(result) << 32) - n1;
    if ((long)(x2 + as_int(b - 1)) >= 0) --result;
    if ((long)(x2 + 0x100000000UL + s) < 0) ++result;

    return result;
}
#endif
