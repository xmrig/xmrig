/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include "crypto/cn/CnAlgo.h"


#ifdef __GNUC__
#   include <x86intrin.h>
#else
#   include <intrin.h>
#   define __restrict__ __restrict
#endif

inline void prep_dv(__m128i* idx, __m128i& v, __m128& n)
{
    v = _mm_load_si128(idx);
    n = _mm_cvtepi32_ps(v);
}

inline __m128 fma_break(__m128 x) 
{ 
    // Break the dependency chain by setitng the exp to ?????01 
    x = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0xFEFFFFFF)), x); 
    return _mm_or_ps(_mm_castsi128_ps(_mm_set1_epi32(0x00800000)), x); 
}

// 14
inline void sub_round(__m128 n0, __m128 n1, __m128 n2, __m128 n3, __m128 rnd_c, __m128& n, __m128& d, __m128& c)
{
    n1 = _mm_add_ps(n1, c);
    __m128 nn = _mm_mul_ps(n0, c);
    nn = _mm_mul_ps(n1, _mm_mul_ps(nn,nn));
    nn = fma_break(nn);
    n = _mm_add_ps(n, nn);

    n3 = _mm_sub_ps(n3, c);
    __m128 dd = _mm_mul_ps(n2, c);
    dd = _mm_mul_ps(n3, _mm_mul_ps(dd,dd));
    dd = fma_break(dd);
    d = _mm_add_ps(d, dd);

    //Constant feedback
    c = _mm_add_ps(c, rnd_c);
    c = _mm_add_ps(c, _mm_set1_ps(0.734375f));
    __m128 r = _mm_add_ps(nn, dd);
    r = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x807FFFFF)), r);
    r = _mm_or_ps(_mm_castsi128_ps(_mm_set1_epi32(0x40000000)), r);
    c = _mm_add_ps(c, r);
}

// 14*8 + 2 = 112
inline void round_compute(__m128 n0, __m128 n1, __m128 n2, __m128 n3, __m128 rnd_c, __m128& c, __m128& r)
{
    __m128 n = _mm_setzero_ps(), d = _mm_setzero_ps();

    sub_round(n0, n1, n2, n3, rnd_c, n, d, c);
    sub_round(n1, n2, n3, n0, rnd_c, n, d, c);
    sub_round(n2, n3, n0, n1, rnd_c, n, d, c);
    sub_round(n3, n0, n1, n2, rnd_c, n, d, c);
    sub_round(n3, n2, n1, n0, rnd_c, n, d, c);
    sub_round(n2, n1, n0, n3, rnd_c, n, d, c);
    sub_round(n1, n0, n3, n2, rnd_c, n, d, c);
    sub_round(n0, n3, n2, n1, rnd_c, n, d, c);

    // Make sure abs(d) > 2.0 - this prevents division by zero and accidental overflows by division by < 1.0
    d = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0xFF7FFFFF)), d);
    d = _mm_or_ps(_mm_castsi128_ps(_mm_set1_epi32(0x40000000)), d);
    r =_mm_add_ps(r, _mm_div_ps(n,d));
}

// 112×4 = 448
template<bool add>
inline __m128i single_compute(__m128 n0, __m128 n1,  __m128 n2,  __m128 n3, float cnt, __m128 rnd_c, __m128& sum)
{
    __m128 c = _mm_set1_ps(cnt);
    __m128 r = _mm_setzero_ps();

    round_compute(n0, n1, n2, n3, rnd_c, c, r);
    round_compute(n0, n1, n2, n3, rnd_c, c, r);
    round_compute(n0, n1, n2, n3, rnd_c, c, r);
    round_compute(n0, n1, n2, n3, rnd_c, c, r);

    // do a quick fmod by setting exp to 2
    r = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x807FFFFF)), r);
    r = _mm_or_ps(_mm_castsi128_ps(_mm_set1_epi32(0x40000000)), r);

    if(add)
        sum = _mm_add_ps(sum, r);
    else
        sum = r;

    r = _mm_mul_ps(r, _mm_set1_ps(536870880.0f)); // 35
    return _mm_cvttps_epi32(r);
}

template<size_t rot>
inline void single_compute_wrap(__m128 n0, __m128 n1, __m128 n2,  __m128 n3, float cnt, __m128 rnd_c, __m128& sum, __m128i& out)
{
    __m128i r = single_compute<rot % 2 != 0>(n0, n1, n2, n3, cnt, rnd_c, sum);
    if(rot != 0)
        r = _mm_or_si128(_mm_slli_si128(r, 16 - rot), _mm_srli_si128(r, rot));
    out = _mm_xor_si128(out, r);
}

template<uint32_t MASK>
inline __m128i* scratchpad_ptr(uint8_t* lpad, uint32_t idx, size_t n) { return reinterpret_cast<__m128i*>(lpad + (idx & MASK) + n*16); }

template<size_t ITER, uint32_t MASK>
void cn_gpu_inner_ssse3(const uint8_t* spad, uint8_t* lpad)
{
    uint32_t s = reinterpret_cast<const uint32_t*>(spad)[0] >> 8;
    __m128i* idx0 = scratchpad_ptr<MASK>(lpad, s, 0);
    __m128i* idx1 = scratchpad_ptr<MASK>(lpad, s, 1);
    __m128i* idx2 = scratchpad_ptr<MASK>(lpad, s, 2);
    __m128i* idx3 = scratchpad_ptr<MASK>(lpad, s, 3);
    __m128 sum0 = _mm_setzero_ps();
    
    for(size_t i = 0; i < ITER; i++)
    {
        __m128 n0, n1, n2, n3;
        __m128i v0, v1, v2, v3;
        __m128 suma, sumb, sum1, sum2, sum3;
        
        prep_dv(idx0, v0, n0);
        prep_dv(idx1, v1, n1);
        prep_dv(idx2, v2, n2);
        prep_dv(idx3, v3, n3);
        __m128 rc = sum0;

        __m128i out, out2;
        out = _mm_setzero_si128();
        single_compute_wrap<0>(n0, n1, n2, n3, 1.3437500f, rc, suma, out);
        single_compute_wrap<1>(n0, n2, n3, n1, 1.2812500f, rc, suma, out);
        single_compute_wrap<2>(n0, n3, n1, n2, 1.3593750f, rc, sumb, out);
        single_compute_wrap<3>(n0, n3, n2, n1, 1.3671875f, rc, sumb, out);
        sum0 = _mm_add_ps(suma, sumb);
        _mm_store_si128(idx0, _mm_xor_si128(v0, out));
        out2 = out;
    
        out = _mm_setzero_si128();
        single_compute_wrap<0>(n1, n0, n2, n3, 1.4296875f, rc, suma, out);
        single_compute_wrap<1>(n1, n2, n3, n0, 1.3984375f, rc, suma, out);
        single_compute_wrap<2>(n1, n3, n0, n2, 1.3828125f, rc, sumb, out);
        single_compute_wrap<3>(n1, n3, n2, n0, 1.3046875f, rc, sumb, out);
        sum1 = _mm_add_ps(suma, sumb);
        _mm_store_si128(idx1, _mm_xor_si128(v1, out));
        out2 = _mm_xor_si128(out2, out);

        out = _mm_setzero_si128();
        single_compute_wrap<0>(n2, n1, n0, n3, 1.4140625f, rc, suma, out);
        single_compute_wrap<1>(n2, n0, n3, n1, 1.2734375f, rc, suma, out);
        single_compute_wrap<2>(n2, n3, n1, n0, 1.2578125f, rc, sumb, out);
        single_compute_wrap<3>(n2, n3, n0, n1, 1.2890625f, rc, sumb, out);
        sum2 = _mm_add_ps(suma, sumb);
        _mm_store_si128(idx2, _mm_xor_si128(v2, out));
        out2 = _mm_xor_si128(out2, out);

        out = _mm_setzero_si128();
        single_compute_wrap<0>(n3, n1, n2, n0, 1.3203125f, rc, suma, out);
        single_compute_wrap<1>(n3, n2, n0, n1, 1.3515625f, rc, suma, out);
        single_compute_wrap<2>(n3, n0, n1, n2, 1.3359375f, rc, sumb, out);
        single_compute_wrap<3>(n3, n0, n2, n1, 1.4609375f, rc, sumb, out);
        sum3 = _mm_add_ps(suma, sumb);
        _mm_store_si128(idx3, _mm_xor_si128(v3, out));
        out2 = _mm_xor_si128(out2, out);
        sum0 = _mm_add_ps(sum0, sum1);
        sum2 = _mm_add_ps(sum2, sum3);
        sum0 = _mm_add_ps(sum0, sum2);

        sum0 = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)), sum0); // take abs(va) by masking the float sign bit
        // vs range 0 - 64 
        n0 = _mm_mul_ps(sum0, _mm_set1_ps(16777216.0f));
        v0 = _mm_cvttps_epi32(n0);
        v0 = _mm_xor_si128(v0, out2);
        v1 = _mm_shuffle_epi32(v0, _MM_SHUFFLE(0, 1, 2, 3));
        v0 = _mm_xor_si128(v0, v1);
        v1 = _mm_shuffle_epi32(v0, _MM_SHUFFLE(0, 1, 0, 1));
        v0 = _mm_xor_si128(v0, v1);

        // vs is now between 0 and 1
        sum0 = _mm_div_ps(sum0, _mm_set1_ps(64.0f));
        uint32_t n = _mm_cvtsi128_si32(v0);
        idx0 = scratchpad_ptr<MASK>(lpad, n, 0);
        idx1 = scratchpad_ptr<MASK>(lpad, n, 1);
        idx2 = scratchpad_ptr<MASK>(lpad, n, 2);
        idx3 = scratchpad_ptr<MASK>(lpad, n, 3);
    }
}

template void cn_gpu_inner_ssse3<xmrig::CnAlgo<xmrig::Algorithm::CN_GPU>().iterations(), xmrig::CnAlgo<xmrig::Algorithm::CN_GPU>().mask()>(const uint8_t* spad, uint8_t* lpad);
