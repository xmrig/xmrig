/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <support@xmrig.com>
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
#ifndef _mm256_bslli_epi128
	#define _mm256_bslli_epi128(a, count) _mm256_slli_si256((a), (count))
#endif
#ifndef _mm256_bsrli_epi128
	#define _mm256_bsrli_epi128(a, count) _mm256_srli_si256((a), (count))
#endif

inline void prep_dv_avx(__m256i* idx, __m256i& v, __m256& n01)
{
    v = _mm256_load_si256(idx);
    n01 = _mm256_cvtepi32_ps(v);
}

inline __m256 fma_break(const __m256& x) 
{ 
    // Break the dependency chain by setitng the exp to ?????01 
    __m256 xx = _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0xFEFFFFFF)), x); 
    return _mm256_or_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x00800000)), xx); 
}

// 14
inline void sub_round(const __m256& n0, const __m256& n1, const __m256& n2, const __m256& n3, const __m256& rnd_c, __m256& n, __m256& d, __m256& c)
{
    __m256 nn = _mm256_mul_ps(n0, c);
    nn = _mm256_mul_ps(_mm256_add_ps(n1, c), _mm256_mul_ps(nn, nn));
    nn = fma_break(nn);
    n = _mm256_add_ps(n, nn);

    __m256 dd = _mm256_mul_ps(n2, c);
    dd = _mm256_mul_ps(_mm256_sub_ps(n3, c), _mm256_mul_ps(dd, dd));
    dd = fma_break(dd);
    d = _mm256_add_ps(d, dd);

    //Constant feedback
    c = _mm256_add_ps(c, rnd_c);
    c = _mm256_add_ps(c, _mm256_set1_ps(0.734375f));
    __m256 r = _mm256_add_ps(nn, dd);
    r = _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x807FFFFF)), r);
    r = _mm256_or_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x40000000)), r);
    c = _mm256_add_ps(c, r);
}

// 14*8 + 2 = 112
inline void round_compute(const __m256& n0, const __m256& n1, const __m256& n2, const __m256& n3, const __m256& rnd_c, __m256& c, __m256& r)
{
    __m256 n = _mm256_setzero_ps(), d = _mm256_setzero_ps();

    sub_round(n0, n1, n2, n3, rnd_c, n, d, c);
    sub_round(n1, n2, n3, n0, rnd_c, n, d, c);
    sub_round(n2, n3, n0, n1, rnd_c, n, d, c);
    sub_round(n3, n0, n1, n2, rnd_c, n, d, c);
    sub_round(n3, n2, n1, n0, rnd_c, n, d, c);
    sub_round(n2, n1, n0, n3, rnd_c, n, d, c);
    sub_round(n1, n0, n3, n2, rnd_c, n, d, c);
    sub_round(n0, n3, n2, n1, rnd_c, n, d, c);

    // Make sure abs(d) > 2.0 - this prevents division by zero and accidental overflows by division by < 1.0
    d = _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0xFF7FFFFF)), d);
    d = _mm256_or_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x40000000)), d);
    r = _mm256_add_ps(r, _mm256_div_ps(n, d));
}

// 112×4 = 448
template <bool add>
inline __m256i double_compute(const __m256& n0, const __m256& n1, const __m256& n2, const __m256& n3,
                              float lcnt, float hcnt, const __m256& rnd_c, __m256& sum)
{
    __m256 c = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(lcnt)), _mm_set1_ps(hcnt), 1);
    __m256 r = _mm256_setzero_ps();

    round_compute(n0, n1, n2, n3, rnd_c, c, r);
    round_compute(n0, n1, n2, n3, rnd_c, c, r);
    round_compute(n0, n1, n2, n3, rnd_c, c, r);
    round_compute(n0, n1, n2, n3, rnd_c, c, r);

    // do a quick fmod by setting exp to 2
    r = _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x807FFFFF)), r);
    r = _mm256_or_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x40000000)), r);

    if(add)
        sum = _mm256_add_ps(sum, r);
    else
        sum = r;

    r = _mm256_mul_ps(r, _mm256_set1_ps(536870880.0f)); // 35
    return _mm256_cvttps_epi32(r);
}

template <size_t rot>
inline void double_compute_wrap(const __m256& n0, const __m256& n1, const __m256& n2, const __m256& n3,
                                float lcnt, float hcnt, const __m256& rnd_c, __m256& sum, __m256i& out)
{
    __m256i r = double_compute<rot % 2 != 0>(n0, n1, n2, n3, lcnt, hcnt, rnd_c, sum);
    if(rot != 0)
        r = _mm256_or_si256(_mm256_bslli_epi128(r, 16 - rot), _mm256_bsrli_epi128(r, rot));

    out = _mm256_xor_si256(out, r);
}

template<uint32_t MASK>
inline __m256i* scratchpad_ptr(uint8_t* lpad, uint32_t idx, size_t n) { return reinterpret_cast<__m256i*>(lpad + (idx & MASK) + n*16); }

template<size_t ITER, uint32_t MASK>
void cn_gpu_inner_avx(const uint8_t* spad, uint8_t* lpad)
{
    uint32_t s = reinterpret_cast<const uint32_t*>(spad)[0] >> 8;
    __m256i* idx0 = scratchpad_ptr<MASK>(lpad, s, 0);
    __m256i* idx2 = scratchpad_ptr<MASK>(lpad, s, 2);
    __m256 sum0 = _mm256_setzero_ps();

    for(size_t i = 0; i < ITER; i++)
    {
        __m256i v01, v23;
        __m256 suma, sumb, sum1;
        __m256 rc = sum0;

        __m256 n01, n23;
        prep_dv_avx(idx0, v01, n01);
        prep_dv_avx(idx2, v23, n23);
        
        __m256i out, out2;
        __m256 n10, n22, n33;
        n10 = _mm256_permute2f128_ps(n01, n01, 0x01);
        n22 = _mm256_permute2f128_ps(n23, n23, 0x00);
        n33 = _mm256_permute2f128_ps(n23, n23, 0x11);
        
        out = _mm256_setzero_si256();
        double_compute_wrap<0>(n01, n10, n22, n33, 1.3437500f, 1.4296875f, rc, suma, out);
        double_compute_wrap<1>(n01, n22, n33, n10, 1.2812500f, 1.3984375f, rc, suma, out);
        double_compute_wrap<2>(n01, n33, n10, n22, 1.3593750f, 1.3828125f, rc, sumb, out);
        double_compute_wrap<3>(n01, n33, n22, n10, 1.3671875f, 1.3046875f, rc, sumb, out);
        _mm256_store_si256(idx0, _mm256_xor_si256(v01, out));
        sum0 = _mm256_add_ps(suma, sumb);
        out2 = out;
        
        __m256 n11, n02, n30;
        n11 = _mm256_permute2f128_ps(n01, n01, 0x11);
        n02 = _mm256_permute2f128_ps(n01, n23, 0x20);
        n30 = _mm256_permute2f128_ps(n01, n23, 0x03);

        out = _mm256_setzero_si256();
        double_compute_wrap<0>(n23, n11, n02, n30, 1.4140625f, 1.3203125f, rc, suma, out);
        double_compute_wrap<1>(n23, n02, n30, n11, 1.2734375f, 1.3515625f, rc, suma, out);
        double_compute_wrap<2>(n23, n30, n11, n02, 1.2578125f, 1.3359375f, rc, sumb, out);
        double_compute_wrap<3>(n23, n30, n02, n11, 1.2890625f, 1.4609375f, rc, sumb, out);
        _mm256_store_si256(idx2, _mm256_xor_si256(v23, out));
        sum1 = _mm256_add_ps(suma, sumb);

        out2 = _mm256_xor_si256(out2, out);
        out2 = _mm256_xor_si256(_mm256_permute2x128_si256(out2,out2,0x41), out2);
        suma = _mm256_permute2f128_ps(sum0, sum1, 0x30);
        sumb = _mm256_permute2f128_ps(sum0, sum1, 0x21);
        sum0 = _mm256_add_ps(suma, sumb);
        sum0 = _mm256_add_ps(sum0, _mm256_permute2f128_ps(sum0, sum0, 0x41));

        // Clear the high 128 bits
        __m128 sum = _mm256_castps256_ps128(sum0);

        sum = _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)), sum); // take abs(va) by masking the float sign bit
        // vs range 0 - 64 
        __m128i v0 = _mm_cvttps_epi32(_mm_mul_ps(sum, _mm_set1_ps(16777216.0f)));
        v0 = _mm_xor_si128(v0, _mm256_castsi256_si128(out2));
        __m128i v1 = _mm_shuffle_epi32(v0, _MM_SHUFFLE(0, 1, 2, 3));
        v0 = _mm_xor_si128(v0, v1);
        v1 = _mm_shuffle_epi32(v0, _MM_SHUFFLE(0, 1, 0, 1));
        v0 = _mm_xor_si128(v0, v1);

        // vs is now between 0 and 1
        sum = _mm_div_ps(sum, _mm_set1_ps(64.0f));
        sum0 = _mm256_insertf128_ps(_mm256_castps128_ps256(sum), sum, 1);
        uint32_t n = _mm_cvtsi128_si32(v0);
        idx0 = scratchpad_ptr<MASK>(lpad, n, 0);
        idx2 = scratchpad_ptr<MASK>(lpad, n, 2);
    }
}

template void cn_gpu_inner_avx<xmrig::CnAlgo<xmrig::Algorithm::CN_GPU>().iterations(), xmrig::CnAlgo<xmrig::Algorithm::CN_GPU>().mask()>(const uint8_t* spad, uint8_t* lpad);
