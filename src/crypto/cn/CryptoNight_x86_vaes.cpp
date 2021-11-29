/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2019 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "CryptoNight_x86_vaes.h"
#include "CryptoNight_monero.h"
#include "CryptoNight.h"


#ifdef __GNUC__
#   include <x86intrin.h>
#if !defined(__clang__) && !defined(__ICC) && __GNUC__ < 10
static inline __m256i
__attribute__((__always_inline__))
  _mm256_loadu2_m128i(const __m128i* const hiaddr, const __m128i* const loaddr)
{
    return _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_loadu_si128(loaddr)), _mm_loadu_si128(hiaddr), 1);
}

static inline void
__attribute__((__always_inline__))
  _mm256_storeu2_m128i(__m128i* const hiaddr, __m128i* const loaddr, const __m256i a)
{
    _mm_storeu_si128(loaddr, _mm256_castsi256_si128(a));
      _mm_storeu_si128(hiaddr, _mm256_extracti128_si256(a, 1));
}
#endif
#else
#   include <intrin.h>
#endif


// This will shift and xor tmp1 into itself as 4 32-bit vals such as
// sl_xor(a1 a2 a3 a4) = a1 (a2^a1) (a3^a2^a1) (a4^a3^a2^a1)
static FORCEINLINE __m128i sl_xor(__m128i tmp1)
{
    __m128i tmp4;
    tmp4 = _mm_slli_si128(tmp1, 0x04);
    tmp1 = _mm_xor_si128(tmp1, tmp4);
    tmp4 = _mm_slli_si128(tmp4, 0x04);
    tmp1 = _mm_xor_si128(tmp1, tmp4);
    tmp4 = _mm_slli_si128(tmp4, 0x04);
    tmp1 = _mm_xor_si128(tmp1, tmp4);
    return tmp1;
}


template<uint8_t rcon>
static FORCEINLINE void aes_genkey_sub(__m128i* xout0, __m128i* xout2)
{
    __m128i xout1 = _mm_aeskeygenassist_si128(*xout2, rcon);
    xout1 = _mm_shuffle_epi32(xout1, 0xFF); // see PSHUFD, set all elems to 4th elem
    *xout0 = sl_xor(*xout0);
    *xout0 = _mm_xor_si128(*xout0, xout1);
    xout1 = _mm_aeskeygenassist_si128(*xout0, 0x00);
    xout1 = _mm_shuffle_epi32(xout1, 0xAA); // see PSHUFD, set all elems to 3rd elem
    *xout2 = sl_xor(*xout2);
    *xout2 = _mm_xor_si128(*xout2, xout1);
}


static NOINLINE void vaes_genkey(const __m128i* memory, __m256i* k0, __m256i* k1, __m256i* k2, __m256i* k3, __m256i* k4, __m256i* k5, __m256i* k6, __m256i* k7, __m256i* k8, __m256i* k9)
{
    __m128i xout0 = _mm_load_si128(memory);
    __m128i xout2 = _mm_load_si128(memory + 1);
    *k0 = _mm256_set_m128i(xout0, xout0);
    *k1 = _mm256_set_m128i(xout2, xout2);

    aes_genkey_sub<0x01>(&xout0, &xout2);
    *k2 = _mm256_set_m128i(xout0, xout0);
    *k3 = _mm256_set_m128i(xout2, xout2);

    aes_genkey_sub<0x02>(&xout0, &xout2);
    *k4 = _mm256_set_m128i(xout0, xout0);
    *k5 = _mm256_set_m128i(xout2, xout2);

    aes_genkey_sub<0x04>(&xout0, &xout2);
    *k6 = _mm256_set_m128i(xout0, xout0);
    *k7 = _mm256_set_m128i(xout2, xout2);

    aes_genkey_sub<0x08>(&xout0, &xout2);
    *k8 = _mm256_set_m128i(xout0, xout0);
    *k9 = _mm256_set_m128i(xout2, xout2);
}


static NOINLINE void vaes_genkey_double(const __m128i* memory1, const __m128i* memory2, __m256i* k0, __m256i* k1, __m256i* k2, __m256i* k3, __m256i* k4, __m256i* k5, __m256i* k6, __m256i* k7, __m256i* k8, __m256i* k9)
{
    __m128i xout0 = _mm_load_si128(memory1);
    __m128i xout1 = _mm_load_si128(memory1 + 1);
    __m128i xout2 = _mm_load_si128(memory2);
    __m128i xout3 = _mm_load_si128(memory2 + 1);
    *k0 = _mm256_set_m128i(xout2, xout0);
    *k1 = _mm256_set_m128i(xout3, xout1);

    aes_genkey_sub<0x01>(&xout0, &xout1);
    aes_genkey_sub<0x01>(&xout2, &xout3);
    *k2 = _mm256_set_m128i(xout2, xout0);
    *k3 = _mm256_set_m128i(xout3, xout1);

    aes_genkey_sub<0x02>(&xout0, &xout1);
    aes_genkey_sub<0x02>(&xout2, &xout3);
    *k4 = _mm256_set_m128i(xout2, xout0);
    *k5 = _mm256_set_m128i(xout3, xout1);

    aes_genkey_sub<0x04>(&xout0, &xout1);
    aes_genkey_sub<0x04>(&xout2, &xout3);
    *k6 = _mm256_set_m128i(xout2, xout0);
    *k7 = _mm256_set_m128i(xout3, xout1);

    aes_genkey_sub<0x08>(&xout0, &xout1);
    aes_genkey_sub<0x08>(&xout2, &xout3);
    *k8 = _mm256_set_m128i(xout2, xout0);
    *k9 = _mm256_set_m128i(xout3, xout1);
}


static FORCEINLINE void vaes_round(__m256i key, __m256i& x01, __m256i& x23, __m256i& x45, __m256i& x67)
{
    x01 = _mm256_aesenc_epi128(x01, key);
    x23 = _mm256_aesenc_epi128(x23, key);
    x45 = _mm256_aesenc_epi128(x45, key);
    x67 = _mm256_aesenc_epi128(x67, key);
}


static FORCEINLINE void vaes_round(__m256i key, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, __m256i& x5, __m256i& x6, __m256i& x7)
{
    x0 = _mm256_aesenc_epi128(x0, key);
    x1 = _mm256_aesenc_epi128(x1, key);
    x2 = _mm256_aesenc_epi128(x2, key);
    x3 = _mm256_aesenc_epi128(x3, key);
    x4 = _mm256_aesenc_epi128(x4, key);
    x5 = _mm256_aesenc_epi128(x5, key);
    x6 = _mm256_aesenc_epi128(x6, key);
    x7 = _mm256_aesenc_epi128(x7, key);
}


namespace xmrig {


template<Algorithm::Id ALGO>
NOINLINE void cn_explode_scratchpad_vaes(cryptonight_ctx* ctx)
{
    constexpr CnAlgo<ALGO> props;

    constexpr size_t N = (props.memory() / sizeof(__m256i)) / (props.half_mem() ? 2 : 1);

    __m256i xin01, xin23, xin45, xin67;
    __m256i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

    const __m128i* input = reinterpret_cast<const __m128i*>(ctx->state);
    __m256i* output = reinterpret_cast<__m256i*>(ctx->memory);

    vaes_genkey(input, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

    if (props.half_mem() && !ctx->first_half) {
        const __m256i* p = reinterpret_cast<const __m256i*>(ctx->save_state);
        xin01 = _mm256_load_si256(p + 0);
        xin23 = _mm256_load_si256(p + 1);
        xin45 = _mm256_load_si256(p + 2);
        xin67 = _mm256_load_si256(p + 3);
    }
    else {
        xin01 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + 4));
        xin23 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + 6));
        xin45 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + 8));
        xin67 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + 10));
    }

    constexpr int output_increment = 64 / sizeof(__m256i);
    constexpr int prefetch_dist = 2048 / sizeof(__m256i);

    __m256i* e = output + N - prefetch_dist;
    __m256i* prefetch_ptr = output + prefetch_dist;

    for (int i = 0; i < 2; ++i) {
        do {
            _mm_prefetch((const char*)(prefetch_ptr), _MM_HINT_T0);
            _mm_prefetch((const char*)(prefetch_ptr + output_increment), _MM_HINT_T0);

            vaes_round(k0, xin01, xin23, xin45, xin67);
            vaes_round(k1, xin01, xin23, xin45, xin67);
            vaes_round(k2, xin01, xin23, xin45, xin67);
            vaes_round(k3, xin01, xin23, xin45, xin67);
            vaes_round(k4, xin01, xin23, xin45, xin67);
            vaes_round(k5, xin01, xin23, xin45, xin67);
            vaes_round(k6, xin01, xin23, xin45, xin67);
            vaes_round(k7, xin01, xin23, xin45, xin67);
            vaes_round(k8, xin01, xin23, xin45, xin67);
            vaes_round(k9, xin01, xin23, xin45, xin67);

            _mm256_store_si256(output + 0, xin01);
            _mm256_store_si256(output + 1, xin23);

            _mm256_store_si256(output + output_increment + 0, xin45);
            _mm256_store_si256(output + output_increment + 1, xin67);

            output += output_increment * 2;
            prefetch_ptr += output_increment * 2;
        } while (output < e);
        e += prefetch_dist;
        prefetch_ptr = output;
    }

    if (props.half_mem() && ctx->first_half) {
        __m256i* p = reinterpret_cast<__m256i*>(ctx->save_state);
        _mm256_store_si256(p + 0, xin01);
        _mm256_store_si256(p + 1, xin23);
        _mm256_store_si256(p + 2, xin45);
        _mm256_store_si256(p + 3, xin67);
    }

    _mm256_zeroupper();
}


template<Algorithm::Id ALGO>
NOINLINE void cn_explode_scratchpad_vaes_double(cryptonight_ctx* ctx1, cryptonight_ctx* ctx2)
{
    constexpr CnAlgo<ALGO> props;

    constexpr size_t N = (props.memory() / sizeof(__m128i)) / (props.half_mem() ? 2 : 1);

    __m256i xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7;
    __m256i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

    const __m128i* input1 = reinterpret_cast<const __m128i*>(ctx1->state);
    const __m128i* input2 = reinterpret_cast<const __m128i*>(ctx2->state);

    __m128i* output1 = reinterpret_cast<__m128i*>(ctx1->memory);
    __m128i* output2 = reinterpret_cast<__m128i*>(ctx2->memory);

    vaes_genkey_double(input1, input2, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

    {
        const bool b = props.half_mem() && !ctx1->first_half && !ctx2->first_half;
        const __m128i* p1 = b ? reinterpret_cast<const __m128i*>(ctx1->save_state) : (input1 + 4);
        const __m128i* p2 = b ? reinterpret_cast<const __m128i*>(ctx2->save_state) : (input2 + 4);
        xin0 = _mm256_loadu2_m128i(p2 + 0, p1 + 0);
        xin1 = _mm256_loadu2_m128i(p2 + 1, p1 + 1);
        xin2 = _mm256_loadu2_m128i(p2 + 2, p1 + 2);
        xin3 = _mm256_loadu2_m128i(p2 + 3, p1 + 3);
        xin4 = _mm256_loadu2_m128i(p2 + 4, p1 + 4);
        xin5 = _mm256_loadu2_m128i(p2 + 5, p1 + 5);
        xin6 = _mm256_loadu2_m128i(p2 + 6, p1 + 6);
        xin7 = _mm256_loadu2_m128i(p2 + 7, p1 + 7);
    }

    constexpr int output_increment = 64 / sizeof(__m128i);
    constexpr int prefetch_dist = 2048 / sizeof(__m128i);

    __m128i* e = output1 + N - prefetch_dist;
    __m128i* prefetch_ptr1 = output1 + prefetch_dist;
    __m128i* prefetch_ptr2 = output2 + prefetch_dist;

    for (int i = 0; i < 2; ++i) {
        do {
            _mm_prefetch((const char*)(prefetch_ptr1), _MM_HINT_T0);
            _mm_prefetch((const char*)(prefetch_ptr1 + output_increment), _MM_HINT_T0);
            _mm_prefetch((const char*)(prefetch_ptr2), _MM_HINT_T0);
            _mm_prefetch((const char*)(prefetch_ptr2 + output_increment), _MM_HINT_T0);

            vaes_round(k0, xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
            vaes_round(k1, xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
            vaes_round(k2, xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
            vaes_round(k3, xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
            vaes_round(k4, xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
            vaes_round(k5, xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
            vaes_round(k6, xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
            vaes_round(k7, xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
            vaes_round(k8, xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);
            vaes_round(k9, xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7);

            _mm256_storeu2_m128i(output2 + 0, output1 + 0, xin0);
            _mm256_storeu2_m128i(output2 + 1, output1 + 1, xin1);
            _mm256_storeu2_m128i(output2 + 2, output1 + 2, xin2);
            _mm256_storeu2_m128i(output2 + 3, output1 + 3, xin3);

            _mm256_storeu2_m128i(output2 + output_increment + 0, output1 + output_increment + 0, xin4);
            _mm256_storeu2_m128i(output2 + output_increment + 1, output1 + output_increment + 1, xin5);
            _mm256_storeu2_m128i(output2 + output_increment + 2, output1 + output_increment + 2, xin6);
            _mm256_storeu2_m128i(output2 + output_increment + 3, output1 + output_increment + 3, xin7);

            output1 += output_increment * 2;
            prefetch_ptr1 += output_increment * 2;
            output2 += output_increment * 2;
            prefetch_ptr2 += output_increment * 2;
        } while (output1 < e);
        e += prefetch_dist;
        prefetch_ptr1 = output1;
        prefetch_ptr2 = output2;
    }

    if (props.half_mem() && ctx1->first_half && ctx2->first_half) {
        __m128i* p1 = reinterpret_cast<__m128i*>(ctx1->save_state);
        __m128i* p2 = reinterpret_cast<__m128i*>(ctx2->save_state);
        _mm256_storeu2_m128i(p2 + 0, p1 + 0, xin0);
        _mm256_storeu2_m128i(p2 + 1, p1 + 1, xin1);
        _mm256_storeu2_m128i(p2 + 2, p1 + 2, xin2);
        _mm256_storeu2_m128i(p2 + 3, p1 + 3, xin3);
        _mm256_storeu2_m128i(p2 + 4, p1 + 4, xin4);
        _mm256_storeu2_m128i(p2 + 5, p1 + 5, xin5);
        _mm256_storeu2_m128i(p2 + 6, p1 + 6, xin6);
        _mm256_storeu2_m128i(p2 + 7, p1 + 7, xin7);
    }

    _mm256_zeroupper();
}


template<Algorithm::Id ALGO>
NOINLINE void cn_implode_scratchpad_vaes(cryptonight_ctx* ctx)
{
    constexpr CnAlgo<ALGO> props;

    constexpr size_t N = (props.memory() / sizeof(__m256i)) / (props.half_mem() ? 2 : 1);

    __m256i xout01, xout23, xout45, xout67;
    __m256i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

    const __m256i* input = reinterpret_cast<const __m256i*>(ctx->memory);
    __m256i* output = reinterpret_cast<__m256i*>(ctx->state);

    vaes_genkey(reinterpret_cast<__m128i*>(output) + 2, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

    xout01 = _mm256_load_si256(output + 2);
    xout23 = _mm256_load_si256(output + 3);
    xout45 = _mm256_load_si256(output + 4);
    xout67 = _mm256_load_si256(output + 5);

    const __m256i* input_begin = input;
    for (size_t part = 0; part < (props.half_mem() ? 2 : 1); ++part) {
        if (props.half_mem() && (part == 1)) {
            input = input_begin;
            ctx->first_half = false;
            cn_explode_scratchpad_vaes<ALGO>(ctx);
        }

        for (size_t i = 0; i < N;) {
            xout01 = _mm256_xor_si256(xout01, input[0]);
            xout23 = _mm256_xor_si256(xout23, input[1]);

            constexpr int input_increment = 64 / sizeof(__m256i);

            xout45 = _mm256_xor_si256(xout45, input[input_increment]);
            xout67 = _mm256_xor_si256(xout67, input[input_increment + 1]);

            input += input_increment * 2;
            i += 4;

            if (i < N) {
                _mm_prefetch((const char*)(input), _MM_HINT_T0);
                _mm_prefetch((const char*)(input + input_increment), _MM_HINT_T0);
            }

            vaes_round(k0, xout01, xout23, xout45, xout67);
            vaes_round(k1, xout01, xout23, xout45, xout67);
            vaes_round(k2, xout01, xout23, xout45, xout67);
            vaes_round(k3, xout01, xout23, xout45, xout67);
            vaes_round(k4, xout01, xout23, xout45, xout67);
            vaes_round(k5, xout01, xout23, xout45, xout67);
            vaes_round(k6, xout01, xout23, xout45, xout67);
            vaes_round(k7, xout01, xout23, xout45, xout67);
            vaes_round(k8, xout01, xout23, xout45, xout67);
            vaes_round(k9, xout01, xout23, xout45, xout67);
        }
    }

    _mm256_store_si256(output + 2, xout01);
    _mm256_store_si256(output + 3, xout23);
    _mm256_store_si256(output + 4, xout45);
    _mm256_store_si256(output + 5, xout67);

    _mm256_zeroupper();
}


template<Algorithm::Id ALGO>
NOINLINE void cn_implode_scratchpad_vaes_double(cryptonight_ctx* ctx1, cryptonight_ctx* ctx2)
{
    constexpr CnAlgo<ALGO> props;

    constexpr size_t N = (props.memory() / sizeof(__m128i)) / (props.half_mem() ? 2 : 1);

    __m256i xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7;
    __m256i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

    const __m128i* input1 = reinterpret_cast<const __m128i*>(ctx1->memory);
    const __m128i* input2 = reinterpret_cast<const __m128i*>(ctx2->memory);

    __m128i* output1 = reinterpret_cast<__m128i*>(ctx1->state);
    __m128i* output2 = reinterpret_cast<__m128i*>(ctx2->state);

    vaes_genkey_double(output1 + 2, output2 + 2, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

    xout0 = _mm256_loadu2_m128i(output2 + 4, output1 + 4);
    xout1 = _mm256_loadu2_m128i(output2 + 5, output1 + 5);
    xout2 = _mm256_loadu2_m128i(output2 + 6, output1 + 6);
    xout3 = _mm256_loadu2_m128i(output2 + 7, output1 + 7);
    xout4 = _mm256_loadu2_m128i(output2 + 8, output1 + 8);
    xout5 = _mm256_loadu2_m128i(output2 + 9, output1 + 9);
    xout6 = _mm256_loadu2_m128i(output2 + 10, output1 + 10);
    xout7 = _mm256_loadu2_m128i(output2 + 11, output1 + 11);

    const __m128i* input_begin1 = input1;
    const __m128i* input_begin2 = input2;
    for (size_t part = 0; part < (props.half_mem() ? 2 : 1); ++part) {
        if (props.half_mem() && (part == 1)) {
            input1 = input_begin1;
            input2 = input_begin2;
            ctx1->first_half = false;
            ctx2->first_half = false;
            cn_explode_scratchpad_vaes_double<ALGO>(ctx1, ctx2);
        }

        for (size_t i = 0; i < N;) {
            xout0 = _mm256_xor_si256(_mm256_loadu2_m128i(input2 + 0, input1 + 0), xout0);
            xout1 = _mm256_xor_si256(_mm256_loadu2_m128i(input2 + 1, input1 + 1), xout1);
            xout2 = _mm256_xor_si256(_mm256_loadu2_m128i(input2 + 2, input1 + 2), xout2);
            xout3 = _mm256_xor_si256(_mm256_loadu2_m128i(input2 + 3, input1 + 3), xout3);

            constexpr int input_increment = 64 / sizeof(__m128i);

            xout4 = _mm256_xor_si256(_mm256_loadu2_m128i(input2 + input_increment + 0, input1 + input_increment + 0), xout4);
            xout5 = _mm256_xor_si256(_mm256_loadu2_m128i(input2 + input_increment + 1, input1 + input_increment + 1), xout5);
            xout6 = _mm256_xor_si256(_mm256_loadu2_m128i(input2 + input_increment + 2, input1 + input_increment + 2), xout6);
            xout7 = _mm256_xor_si256(_mm256_loadu2_m128i(input2 + input_increment + 3, input1 + input_increment + 3), xout7);

            input1 += input_increment * 2;
            input2 += input_increment * 2;
            i += 8;

            if (i < N) {
                _mm_prefetch((const char*)(input1), _MM_HINT_T0);
                _mm_prefetch((const char*)(input1 + input_increment), _MM_HINT_T0);
                _mm_prefetch((const char*)(input2), _MM_HINT_T0);
                _mm_prefetch((const char*)(input2 + input_increment), _MM_HINT_T0);
            }

            vaes_round(k0, xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
            vaes_round(k1, xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
            vaes_round(k2, xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
            vaes_round(k3, xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
            vaes_round(k4, xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
            vaes_round(k5, xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
            vaes_round(k6, xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
            vaes_round(k7, xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
            vaes_round(k8, xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
            vaes_round(k9, xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7);
        }
    }

    _mm256_storeu2_m128i(output2 + 4, output1 + 4, xout0);
    _mm256_storeu2_m128i(output2 + 5, output1 + 5, xout1);
    _mm256_storeu2_m128i(output2 + 6, output1 + 6, xout2);
    _mm256_storeu2_m128i(output2 + 7, output1 + 7, xout3);
    _mm256_storeu2_m128i(output2 + 8, output1 + 8, xout4);
    _mm256_storeu2_m128i(output2 + 9, output1 + 9, xout5);
    _mm256_storeu2_m128i(output2 + 10, output1 + 10, xout6);
    _mm256_storeu2_m128i(output2 + 11, output1 + 11, xout7);

    _mm256_zeroupper();
}


template<Algorithm::Id ALGO>
void VAES_Instance()
{
    cn_explode_scratchpad_vaes<ALGO>(nullptr);
    cn_explode_scratchpad_vaes_double<ALGO>(nullptr, nullptr);
    cn_implode_scratchpad_vaes<ALGO>(nullptr);
    cn_implode_scratchpad_vaes_double<ALGO>(nullptr, nullptr);
}


void (*vaes_instances[])() = {
    VAES_Instance<Algorithm::CN_0>,
    VAES_Instance<Algorithm::CN_1>,
    VAES_Instance<Algorithm::CN_2>,
    VAES_Instance<Algorithm::CN_R>,
    VAES_Instance<Algorithm::CN_FAST>,
    VAES_Instance<Algorithm::CN_HALF>,
    VAES_Instance<Algorithm::CN_XAO>,
    VAES_Instance<Algorithm::CN_RTO>,
    VAES_Instance<Algorithm::CN_RWZ>,
    VAES_Instance<Algorithm::CN_ZLS>,
    VAES_Instance<Algorithm::CN_DOUBLE>,
    VAES_Instance<Algorithm::CN_CCX>,
    VAES_Instance<Algorithm::CN_LITE_0>,
    VAES_Instance<Algorithm::CN_LITE_1>,
    VAES_Instance<Algorithm::CN_HEAVY_0>,
    VAES_Instance<Algorithm::CN_HEAVY_TUBE>,
    VAES_Instance<Algorithm::CN_HEAVY_XHV>,
    VAES_Instance<Algorithm::CN_PICO_0>,
    VAES_Instance<Algorithm::CN_PICO_TLO>,
    VAES_Instance<Algorithm::CN_UPX2>,
    VAES_Instance<Algorithm::CN_GR_0>,
    VAES_Instance<Algorithm::CN_GR_1>,
    VAES_Instance<Algorithm::CN_GR_2>,
    VAES_Instance<Algorithm::CN_GR_3>,
    VAES_Instance<Algorithm::CN_GR_4>,
    VAES_Instance<Algorithm::CN_GR_5>,
    VAES_Instance<Algorithm::CN_GPU>,
};


} // xmrig
