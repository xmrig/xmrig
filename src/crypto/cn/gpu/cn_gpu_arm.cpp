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


#include <arm_neon.h>


#include "crypto/cn/CnAlgo.h"


inline void vandq_f32(float32x4_t &v, uint32_t v2)
{
    uint32x4_t vc = vdupq_n_u32(v2);
    v = (float32x4_t)vandq_u32((uint32x4_t)v, vc);
}


inline void vorq_f32(float32x4_t &v, uint32_t v2)
{
    uint32x4_t vc = vdupq_n_u32(v2);
    v = (float32x4_t)vorrq_u32((uint32x4_t)v, vc);
}


template <size_t v>
inline void vrot_si32(int32x4_t &r)
{
    r = (int32x4_t)vextq_s8((int8x16_t)r, (int8x16_t)r, v);
}

template <>
inline void vrot_si32<0>(int32x4_t &r)
{
}


inline uint32_t vheor_s32(const int32x4_t &v)
{
    int32x4_t v0 = veorq_s32(v, vrev64q_s32(v));
    int32x2_t vf = veor_s32(vget_high_s32(v0), vget_low_s32(v0));
    return (uint32_t)vget_lane_s32(vf, 0);
}


inline void prep_dv(int32_t *idx, int32x4_t &v, float32x4_t &n)
{
    v = vld1q_s32(idx);
    n = vcvtq_f32_s32(v);
}


inline void sub_round(const float32x4_t &n0, const float32x4_t &n1, const float32x4_t &n2, const float32x4_t &n3, const float32x4_t &rnd_c, float32x4_t &n, float32x4_t &d, float32x4_t &c)
{
    float32x4_t ln1 = vaddq_f32(n1, c);
    float32x4_t nn = vmulq_f32(n0, c);
    nn = vmulq_f32(ln1, vmulq_f32(nn, nn));
    vandq_f32(nn, 0xFEFFFFFF);
    vorq_f32(nn, 0x00800000);
    n = vaddq_f32(n, nn);

    float32x4_t ln3 = vsubq_f32(n3, c);
    float32x4_t dd = vmulq_f32(n2, c);
    dd = vmulq_f32(ln3, vmulq_f32(dd, dd));
    vandq_f32(dd, 0xFEFFFFFF);
    vorq_f32(dd, 0x00800000);
    d = vaddq_f32(d, dd);

    //Constant feedback
    c = vaddq_f32(c, rnd_c);
    c = vaddq_f32(c, vdupq_n_f32(0.734375f));
    float32x4_t r = vaddq_f32(nn, dd);
    vandq_f32(r, 0x807FFFFF);
    vorq_f32(r, 0x40000000);
    c = vaddq_f32(c, r);
}


inline void round_compute(const float32x4_t &n0, const float32x4_t &n1, const float32x4_t &n2, const float32x4_t &n3, const float32x4_t &rnd_c, float32x4_t &c, float32x4_t &r)
{
    float32x4_t n = vdupq_n_f32(0.0f), d = vdupq_n_f32(0.0f);

    sub_round(n0, n1, n2, n3, rnd_c, n, d, c);
    sub_round(n1, n2, n3, n0, rnd_c, n, d, c);
    sub_round(n2, n3, n0, n1, rnd_c, n, d, c);
    sub_round(n3, n0, n1, n2, rnd_c, n, d, c);
    sub_round(n3, n2, n1, n0, rnd_c, n, d, c);
    sub_round(n2, n1, n0, n3, rnd_c, n, d, c);
    sub_round(n1, n0, n3, n2, rnd_c, n, d, c);
    sub_round(n0, n3, n2, n1, rnd_c, n, d, c);

    // Make sure abs(d) > 2.0 - this prevents division by zero and accidental overflows by division by < 1.0
    vandq_f32(d, 0xFF7FFFFF);
    vorq_f32(d, 0x40000000);
    r = vaddq_f32(r, vdivq_f32(n, d));
}


// 112×4 = 448
template <bool add>
inline int32x4_t single_compute(const float32x4_t &n0, const float32x4_t &n1, const float32x4_t &n2, const float32x4_t &n3, float cnt, const float32x4_t &rnd_c, float32x4_t &sum)
{
    float32x4_t c = vdupq_n_f32(cnt);
    float32x4_t r = vdupq_n_f32(0.0f);

    round_compute(n0, n1, n2, n3, rnd_c, c, r);
    round_compute(n0, n1, n2, n3, rnd_c, c, r);
    round_compute(n0, n1, n2, n3, rnd_c, c, r);
    round_compute(n0, n1, n2, n3, rnd_c, c, r);

    // do a quick fmod by setting exp to 2
    vandq_f32(r, 0x807FFFFF);
    vorq_f32(r, 0x40000000);

    if (add) {
        sum = vaddq_f32(sum, r);
    } else {
        sum = r;
    }

    const float32x4_t cc2 = vdupq_n_f32(536870880.0f);
    r = vmulq_f32(r, cc2); // 35
    return vcvtq_s32_f32(r);
}


template<size_t rot>
inline void single_compute_wrap(const float32x4_t &n0, const float32x4_t &n1, const float32x4_t &n2, const float32x4_t &n3, float cnt, const float32x4_t &rnd_c, float32x4_t &sum, int32x4_t &out)
{
    int32x4_t r = single_compute<rot % 2 != 0>(n0, n1, n2, n3, cnt, rnd_c, sum);
    vrot_si32<rot>(r);
    out = veorq_s32(out, r);
}


template<uint32_t MASK>
inline int32_t *scratchpad_ptr(uint8_t* lpad, uint32_t idx, size_t n) { return reinterpret_cast<int32_t *>(lpad + (idx & MASK) + n * 16); }


template<size_t ITER, uint32_t MASK>
void cn_gpu_inner_arm(const uint8_t *spad, uint8_t *lpad)
{
    uint32_t s = reinterpret_cast<const uint32_t*>(spad)[0] >> 8;
    int32_t *idx0 = scratchpad_ptr<MASK>(lpad, s, 0);
    int32_t *idx1 = scratchpad_ptr<MASK>(lpad, s, 1);
    int32_t *idx2 = scratchpad_ptr<MASK>(lpad, s, 2);
    int32_t *idx3 = scratchpad_ptr<MASK>(lpad, s, 3);
    float32x4_t sum0 = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < ITER; i++) {
        float32x4_t n0, n1, n2, n3;
        int32x4_t v0, v1, v2, v3;
        float32x4_t suma, sumb, sum1, sum2, sum3;

        prep_dv(idx0, v0, n0);
        prep_dv(idx1, v1, n1);
        prep_dv(idx2, v2, n2);
        prep_dv(idx3, v3, n3);
        float32x4_t rc = sum0;

        int32x4_t out, out2;
        out = vdupq_n_s32(0);
        single_compute_wrap<0>(n0, n1, n2, n3, 1.3437500f, rc, suma, out);
        single_compute_wrap<1>(n0, n2, n3, n1, 1.2812500f, rc, suma, out);
        single_compute_wrap<2>(n0, n3, n1, n2, 1.3593750f, rc, sumb, out);
        single_compute_wrap<3>(n0, n3, n2, n1, 1.3671875f, rc, sumb, out);
        sum0 = vaddq_f32(suma, sumb);
        vst1q_s32(idx0, veorq_s32(v0, out));
        out2 = out;

        out = vdupq_n_s32(0);
        single_compute_wrap<0>(n1, n0, n2, n3, 1.4296875f, rc, suma, out);
        single_compute_wrap<1>(n1, n2, n3, n0, 1.3984375f, rc, suma, out);
        single_compute_wrap<2>(n1, n3, n0, n2, 1.3828125f, rc, sumb, out);
        single_compute_wrap<3>(n1, n3, n2, n0, 1.3046875f, rc, sumb, out);
        sum1 = vaddq_f32(suma, sumb);
        vst1q_s32(idx1, veorq_s32(v1, out));
        out2 = veorq_s32(out2, out);

        out = vdupq_n_s32(0);
        single_compute_wrap<0>(n2, n1, n0, n3, 1.4140625f, rc, suma, out);
        single_compute_wrap<1>(n2, n0, n3, n1, 1.2734375f, rc, suma, out);
        single_compute_wrap<2>(n2, n3, n1, n0, 1.2578125f, rc, sumb, out);
        single_compute_wrap<3>(n2, n3, n0, n1, 1.2890625f, rc, sumb, out);
        sum2 = vaddq_f32(suma, sumb);
        vst1q_s32(idx2, veorq_s32(v2, out));
        out2 = veorq_s32(out2, out);

        out = vdupq_n_s32(0);
        single_compute_wrap<0>(n3, n1, n2, n0, 1.3203125f, rc, suma, out);
        single_compute_wrap<1>(n3, n2, n0, n1, 1.3515625f, rc, suma, out);
        single_compute_wrap<2>(n3, n0, n1, n2, 1.3359375f, rc, sumb, out);
        single_compute_wrap<3>(n3, n0, n2, n1, 1.4609375f, rc, sumb, out);
        sum3 = vaddq_f32(suma, sumb);
        vst1q_s32(idx3, veorq_s32(v3, out));
        out2 = veorq_s32(out2, out);

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);

        const float32x4_t cc1 = vdupq_n_f32(16777216.0f);
        const float32x4_t cc2 = vdupq_n_f32(64.0f);
        vandq_f32(sum0, 0x7fffffff); // take abs(va) by masking the float sign bit
        // vs range 0 - 64
        n0 = vmulq_f32(sum0, cc1);
        v0 = vcvtq_s32_f32(n0);
        v0 = veorq_s32(v0, out2);
        uint32_t n = vheor_s32(v0);

        // vs is now between 0 and 1
        sum0 = vdivq_f32(sum0, cc2);
        idx0 = scratchpad_ptr<MASK>(lpad, n, 0);
        idx1 = scratchpad_ptr<MASK>(lpad, n, 1);
        idx2 = scratchpad_ptr<MASK>(lpad, n, 2);
        idx3 = scratchpad_ptr<MASK>(lpad, n, 3);
    }
}

template void cn_gpu_inner_arm<xmrig::CnAlgo<xmrig::Algorithm::CN_GPU>().iterations(), xmrig::CnAlgo<xmrig::Algorithm::CN_GPU>().mask()>(const uint8_t* spad, uint8_t* lpad);
