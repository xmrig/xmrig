#include "log/Log.h"

#ifndef __CRYPTONIGHT_X86_H__
#define __CRYPTONIGHT_X86_H__


#ifdef __GNUC__
#   include <x86intrin.h>
#else
#   include <intrin.h>
#   define __restrict__ __restrict
#endif


#include "crypto/CryptoNight.h"
#include "crypto/soft_aes.h"


extern "C"
{
#include "crypto/c_keccak.h"
#include "crypto/c_groestl.h"
#include "crypto/c_blake256.h"
#include "crypto/c_jh.h"
#include "crypto/c_skein.h"
}


static inline void do_blake_hash(const void* input, size_t len, char* output) {
    blake256_hash(reinterpret_cast<uint8_t*>(output), static_cast<const uint8_t*>(input), len);
}


static inline void do_groestl_hash(const void* input, size_t len, char* output) {
    groestl(static_cast<const uint8_t*>(input), len * 8, reinterpret_cast<uint8_t*>(output));
}


static inline void do_jh_hash(const void* input, size_t len, char* output) {
    jh_hash(32 * 8, static_cast<const uint8_t*>(input), 8 * len, reinterpret_cast<uint8_t*>(output));
}


static inline void do_skein_hash(const void* input, size_t len, char* output) {
    xmr_skein(static_cast<const uint8_t*>(input), reinterpret_cast<uint8_t*>(output));
}


void (* const extra_hashes[4])(const void *, size_t, char *) = {do_blake_hash, do_groestl_hash, do_jh_hash, do_skein_hash};



#if defined(__x86_64__) || defined(_M_AMD64)
#   define EXTRACT64(X) _mm_cvtsi128_si64(X)

#   ifdef __GNUC__
static inline uint64_t __umul128(uint64_t a, uint64_t b, uint64_t* hi)
{
    unsigned __int128 r = (unsigned __int128) a * (unsigned __int128) b;
    *hi = r >> 64;
    return (uint64_t) r;
}
#   else
    #define __umul128 _umul128
#   endif
#elif defined(__i386__) || defined(_M_IX86)
#   define HI32(X) \
    _mm_srli_si128((X), 4)


#   define EXTRACT64(X) \
    ((uint64_t)(uint32_t)_mm_cvtsi128_si32(X) | \
    ((uint64_t)(uint32_t)_mm_cvtsi128_si32(HI32(X)) << 32))

static inline uint64_t __umul128(uint64_t multiplier, uint64_t multiplicand, uint64_t *product_hi) {
    // multiplier   = ab = a * 2^32 + b
    // multiplicand = cd = c * 2^32 + d
    // ab * cd = a * c * 2^64 + (a * d + b * c) * 2^32 + b * d
    uint64_t a = multiplier >> 32;
    uint64_t b = multiplier & 0xFFFFFFFF;
    uint64_t c = multiplicand >> 32;
    uint64_t d = multiplicand & 0xFFFFFFFF;

    //uint64_t ac = a * c;
    uint64_t ad = a * d;
    //uint64_t bc = b * c;
    uint64_t bd = b * d;

    uint64_t adbc = ad + (b * c);
    uint64_t adbc_carry = adbc < ad ? 1 : 0;

    // multiplier * multiplicand = product_hi * 2^64 + product_lo
    uint64_t product_lo = bd + (adbc << 32);
    uint64_t product_lo_carry = product_lo < bd ? 1 : 0;
    *product_hi = (a * c) + (adbc >> 32) + (adbc_carry << 32) + product_lo_carry;

    return product_lo;
}
#endif


// This will shift and xor tmp1 into itself as 4 32-bit vals such as
// sl_xor(a1 a2 a3 a4) = a1 (a2^a1) (a3^a2^a1) (a4^a3^a2^a1)
static inline __m128i sl_xor(__m128i tmp1)
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
static inline void aes_genkey_sub(__m128i* xout0, __m128i* xout2)
{
    __m128i xout1 = _mm_aeskeygenassist_si128(*xout2, rcon);
    xout1  = _mm_shuffle_epi32(xout1, 0xFF); // see PSHUFD, set all elems to 4th elem
    *xout0 = sl_xor(*xout0);
    *xout0 = _mm_xor_si128(*xout0, xout1);
    xout1  = _mm_aeskeygenassist_si128(*xout0, 0x00);
    xout1  = _mm_shuffle_epi32(xout1, 0xAA); // see PSHUFD, set all elems to 3rd elem
    *xout2 = sl_xor(*xout2);
    *xout2 = _mm_xor_si128(*xout2, xout1);
}


template<uint8_t rcon>
static inline void soft_aes_genkey_sub(__m128i* xout0, __m128i* xout2)
{
    __m128i xout1 = soft_aeskeygenassist<rcon>(*xout2);
    xout1  = _mm_shuffle_epi32(xout1, 0xFF); // see PSHUFD, set all elems to 4th elem
    *xout0 = sl_xor(*xout0);
    *xout0 = _mm_xor_si128(*xout0, xout1);
    xout1  = soft_aeskeygenassist<0x00>(*xout0);
    xout1  = _mm_shuffle_epi32(xout1, 0xAA); // see PSHUFD, set all elems to 3rd elem
    *xout2 = sl_xor(*xout2);
    *xout2 = _mm_xor_si128(*xout2, xout1);
}


template<bool SOFT_AES>
static inline void aes_genkey(const __m128i* memory, __m128i* k0, __m128i* k1, __m128i* k2, __m128i* k3, __m128i* k4, __m128i* k5, __m128i* k6, __m128i* k7, __m128i* k8, __m128i* k9)
{
    __m128i xout0 = _mm_load_si128(memory);
    __m128i xout2 = _mm_load_si128(memory + 1);
    *k0 = xout0;
    *k1 = xout2;

    SOFT_AES ? soft_aes_genkey_sub<0x01>(&xout0, &xout2) : aes_genkey_sub<0x01>(&xout0, &xout2);
    *k2 = xout0;
    *k3 = xout2;

    SOFT_AES ? soft_aes_genkey_sub<0x02>(&xout0, &xout2) : aes_genkey_sub<0x02>(&xout0, &xout2);
    *k4 = xout0;
    *k5 = xout2;

    SOFT_AES ? soft_aes_genkey_sub<0x04>(&xout0, &xout2) : aes_genkey_sub<0x04>(&xout0, &xout2);
    *k6 = xout0;
    *k7 = xout2;

    SOFT_AES ? soft_aes_genkey_sub<0x08>(&xout0, &xout2) : aes_genkey_sub<0x08>(&xout0, &xout2);
    *k8 = xout0;
    *k9 = xout2;
}


template<bool SOFT_AES>
static inline void aes_round(__m128i key, __m128i* x0, __m128i* x1, __m128i* x2, __m128i* x3, __m128i* x4, __m128i* x5, __m128i* x6, __m128i* x7)
{
    if (SOFT_AES) {
        *x0 = soft_aesenc(*x0, key);
        *x1 = soft_aesenc(*x1, key);
        *x2 = soft_aesenc(*x2, key);
        *x3 = soft_aesenc(*x3, key);
        *x4 = soft_aesenc(*x4, key);
        *x5 = soft_aesenc(*x5, key);
        *x6 = soft_aesenc(*x6, key);
        *x7 = soft_aesenc(*x7, key);
    }
    else {
        *x0 = _mm_aesenc_si128(*x0, key);
        *x1 = _mm_aesenc_si128(*x1, key);
        *x2 = _mm_aesenc_si128(*x2, key);
        *x3 = _mm_aesenc_si128(*x3, key);
        *x4 = _mm_aesenc_si128(*x4, key);
        *x5 = _mm_aesenc_si128(*x5, key);
        *x6 = _mm_aesenc_si128(*x6, key);
        *x7 = _mm_aesenc_si128(*x7, key);
    }
}


template<size_t MEM, bool SOFT_AES>
static inline void cn_explode_scratchpad(const __m128i *input, __m128i *output)
{
    __m128i xin0, xin1, xin2, xin3, xin4, xin5, xin6, xin7;
    __m128i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

    _mm_prefetch((const char*)input, _MM_HINT_NTA);
    _mm_prefetch((const char*)input + 8, _MM_HINT_NTA);

    aes_genkey<SOFT_AES>(input, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

    xin0 = _mm_load_si128(input + 4);
    xin1 = _mm_load_si128(input + 5);
    xin2 = _mm_load_si128(input + 6);
    xin3 = _mm_load_si128(input + 7);
    xin4 = _mm_load_si128(input + 8);
    xin5 = _mm_load_si128(input + 9);
    xin6 = _mm_load_si128(input + 10);
    xin7 = _mm_load_si128(input + 11);

    for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8) {
        _mm_prefetch((const char*)output + i + 8, _MM_HINT_T2);
        
        aes_round<SOFT_AES>(k0, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k1, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k2, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k3, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k4, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k5, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k6, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k7, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k8, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);
        aes_round<SOFT_AES>(k9, &xin0, &xin1, &xin2, &xin3, &xin4, &xin5, &xin6, &xin7);

        _mm_store_si128(output + i + 0, xin0);
        _mm_store_si128(output + i + 1, xin1);
        _mm_store_si128(output + i + 2, xin2);
        _mm_store_si128(output + i + 3, xin3);
        _mm_store_si128(output + i + 4, xin4);
        _mm_store_si128(output + i + 5, xin5);
        _mm_store_si128(output + i + 6, xin6);
        _mm_store_si128(output + i + 7, xin7);
    }
}


template<size_t MEM, bool SOFT_AES>
static inline void cn_implode_scratchpad(const __m128i *input, __m128i *output)
{
    __m128i xout0, xout1, xout2, xout3, xout4, xout5, xout6, xout7;
    __m128i k0, k1, k2, k3, k4, k5, k6, k7, k8, k9;

    _mm_prefetch((const char*)output, _MM_HINT_NTA);
    _mm_prefetch((const char*)input, _MM_HINT_NTA);

    aes_genkey<SOFT_AES>(output + 2, &k0, &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9);

    xout0 = _mm_load_si128(output + 4);
    xout1 = _mm_load_si128(output + 5);
    xout2 = _mm_load_si128(output + 6);
    xout3 = _mm_load_si128(output + 7);
    xout4 = _mm_load_si128(output + 8);
    xout5 = _mm_load_si128(output + 9);
    xout6 = _mm_load_si128(output + 10);
    xout7 = _mm_load_si128(output + 11);

    for (size_t i = 0; i < MEM / sizeof(__m128i); i += 8)
    {
        _mm_prefetch((const char*)input + i + 8, _MM_HINT_NTA);

        xout0 = _mm_xor_si128(_mm_load_si128(input + i + 0), xout0);
        xout1 = _mm_xor_si128(_mm_load_si128(input + i + 1), xout1);
        xout2 = _mm_xor_si128(_mm_load_si128(input + i + 2), xout2);
        xout3 = _mm_xor_si128(_mm_load_si128(input + i + 3), xout3);
        xout4 = _mm_xor_si128(_mm_load_si128(input + i + 4), xout4);
        xout5 = _mm_xor_si128(_mm_load_si128(input + i + 5), xout5);
        xout6 = _mm_xor_si128(_mm_load_si128(input + i + 6), xout6);
        xout7 = _mm_xor_si128(_mm_load_si128(input + i + 7), xout7);

        aes_round<SOFT_AES>(k0, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k1, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k2, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k3, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k4, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k5, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k6, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k7, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k8, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
        aes_round<SOFT_AES>(k9, &xout0, &xout1, &xout2, &xout3, &xout4, &xout5, &xout6, &xout7);
    }

    _mm_store_si128(output + 4, xout0);
    _mm_store_si128(output + 5, xout1);
    _mm_store_si128(output + 6, xout2);
    _mm_store_si128(output + 7, xout3);
    _mm_store_si128(output + 8, xout4);
    _mm_store_si128(output + 9, xout5);
    _mm_store_si128(output + 10, xout6);
    _mm_store_si128(output + 11, xout7);
}


template<size_t ITERATIONS, size_t MEM, size_t MASK, bool SOFT_AES>
inline void cryptonight_hash(const void *__restrict__ input, size_t size, void *__restrict__ output, cryptonight_ctx *__restrict__ ctx)
{
    keccak(static_cast<const uint8_t*>(input), (int) size, ctx->state0, 200);

    cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) ctx->state0, (__m128i*) ctx->memory);

    const uint8_t* l0 = ctx->memory;
    uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state0);

    uint64_t al0 = h0[0] ^ h0[4];
    uint64_t ah0 = h0[1] ^ h0[5];
    __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);

    uint64_t idx0 = h0[0] ^ h0[4];

    for (size_t i = 0; i < ITERATIONS; i++) {
        __m128i cx;
        cx = _mm_load_si128((__m128i *) &l0[idx0 & MASK]);

        if (SOFT_AES) {
            cx = soft_aesenc(cx, _mm_set_epi64x(ah0, al0));
        }
        else {
            cx = _mm_aesenc_si128(cx, _mm_set_epi64x(ah0, al0));
        }

        _mm_store_si128((__m128i *) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx));
        idx0 = EXTRACT64(cx);
        bx0 = cx;

        uint64_t hi, lo, cl, ch;
        cl = ((uint64_t*) &l0[idx0 & MASK])[0];
        ch = ((uint64_t*) &l0[idx0 & MASK])[1];
        lo = __umul128(idx0, cl, &hi);

        al0 += hi;
        ah0 += lo;

        ((uint64_t*)&l0[idx0 & MASK])[0] = al0;
        ((uint64_t*)&l0[idx0 & MASK])[1] = ah0;

        ah0 ^= ch;
        al0 ^= cl;
        idx0 = al0;
    }

    cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) ctx->memory, (__m128i*) ctx->state0);

    keccakf(h0, 24);
    extra_hashes[ctx->state0[0] & 3](ctx->state0, 200, static_cast<char*>(output));
}


template<size_t ITERATIONS, size_t MEM, size_t MASK, bool SOFT_AES>
inline void cryptonight_double_hash(const void *__restrict__ input, size_t size, void *__restrict__ output, struct cryptonight_ctx *__restrict__ ctx)
{
    keccak((const uint8_t *) input,        (int) size, ctx->state0, 200);
    keccak((const uint8_t *) input + size, (int) size, ctx->state1, 200);

    const uint8_t* l0 = ctx->memory;
    const uint8_t* l1 = ctx->memory + MEM;
    uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state0);
    uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state1);

    cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
    cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);

    uint64_t al0 = h0[0] ^ h0[4];
    uint64_t al1 = h1[0] ^ h1[4];
    uint64_t ah0 = h0[1] ^ h0[5];
    uint64_t ah1 = h1[1] ^ h1[5];

    __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
    __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);

    uint64_t idx0 = h0[0] ^ h0[4];
    uint64_t idx1 = h1[0] ^ h1[4];

    for (size_t i = 0; i < ITERATIONS; i++) {
        __m128i cx0 = _mm_load_si128((__m128i *) &l0[idx0 & MASK]);
        __m128i cx1 = _mm_load_si128((__m128i *) &l1[idx1 & MASK]);

        if (SOFT_AES) {
            cx0 = soft_aesenc(cx0, _mm_set_epi64x(ah0, al0));
            cx1 = soft_aesenc(cx1, _mm_set_epi64x(ah1, al1));
        }
        else {
            cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
            cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
        }

        _mm_store_si128((__m128i *) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
        _mm_store_si128((__m128i *) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));

        idx0 = EXTRACT64(cx0);
        idx1 = EXTRACT64(cx1);

        bx0 = cx0;
        bx1 = cx1;

        uint64_t hi, lo, cl, ch;
        cl = ((uint64_t*) &l0[idx0 & MASK])[0];
        ch = ((uint64_t*) &l0[idx0 & MASK])[1];
        lo = __umul128(idx0, cl, &hi);

        al0 += hi;
        ah0 += lo;

        ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
        ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

        ah0 ^= ch;
        al0 ^= cl;
        idx0 = al0;

        cl = ((uint64_t*) &l1[idx1 & MASK])[0];
        ch = ((uint64_t*) &l1[idx1 & MASK])[1];
        lo = __umul128(idx1, cl, &hi);

        al1 += hi;
        ah1 += lo;

        ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
        ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

        ah1 ^= ch;
        al1 ^= cl;
        idx1 = al1;
    }

    cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
    cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);

    keccakf(h0, 24);
    keccakf(h1, 24);

    extra_hashes[ctx->state0[0] & 3](ctx->state0, 200, static_cast<char*>(output));
    extra_hashes[ctx->state1[0] & 3](ctx->state1, 200, static_cast<char*>(output) + 32);
}

template<size_t ITERATIONS, size_t MEM, size_t MASK, bool SOFT_AES>
inline void cryptonight_triple_hash(const void *__restrict__ input, size_t size, void *__restrict__ output, struct cryptonight_ctx *__restrict__ ctx)
{
    keccak((const uint8_t *) input,        (int) size, ctx->state0, 200);
    keccak((const uint8_t *) input + size, (int) size, ctx->state1, 200);
    keccak((const uint8_t *) input + size + size, (int) size, ctx->state2, 200);

    const uint8_t* l0 = ctx->memory;
    const uint8_t* l1 = ctx->memory + MEM;
    const uint8_t* l2 = ctx->memory + MEM + MEM;
    uint64_t* h0 = reinterpret_cast<uint64_t*>(ctx->state0);
    uint64_t* h1 = reinterpret_cast<uint64_t*>(ctx->state1);
    uint64_t* h2 = reinterpret_cast<uint64_t*>(ctx->state2);
    cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h0, (__m128i*) l0);
    cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h1, (__m128i*) l1);
    cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) h2, (__m128i*) l2);
    uint64_t al0 = h0[0] ^ h0[4];
    uint64_t al1 = h1[0] ^ h1[4];
    uint64_t al2 = h2[0] ^ h2[4];
    uint64_t ah0 = h0[1] ^ h0[5];
    uint64_t ah1 = h1[1] ^ h1[5];
    uint64_t ah2 = h2[1] ^ h2[5];
    __m128i bx0 = _mm_set_epi64x(h0[3] ^ h0[7], h0[2] ^ h0[6]);
    __m128i bx1 = _mm_set_epi64x(h1[3] ^ h1[7], h1[2] ^ h1[6]);
    __m128i bx2 = _mm_set_epi64x(h2[3] ^ h2[7], h2[2] ^ h2[6]);

    uint64_t idx0 = h0[0] ^ h0[4];
    uint64_t idx1 = h1[0] ^ h1[4];
    uint64_t idx2 = h2[0] ^ h2[4];
    for (size_t i = 0; i < ITERATIONS; i++) {
        __m128i cx0 = _mm_load_si128((__m128i *) &l0[idx0 & MASK]);
        __m128i cx1 = _mm_load_si128((__m128i *) &l1[idx1 & MASK]);
        __m128i cx2 = _mm_load_si128((__m128i *) &l2[idx2 & MASK]);

        if (SOFT_AES) {
            cx0 = soft_aesenc(cx0, _mm_set_epi64x(ah0, al0));
            cx1 = soft_aesenc(cx1, _mm_set_epi64x(ah1, al1));
            cx2 = soft_aesenc(cx2, _mm_set_epi64x(ah2, al2));
        }
        else {
            cx0 = _mm_aesenc_si128(cx0, _mm_set_epi64x(ah0, al0));
            cx1 = _mm_aesenc_si128(cx1, _mm_set_epi64x(ah1, al1));
            cx2 = _mm_aesenc_si128(cx2, _mm_set_epi64x(ah2, al2));
        }

        _mm_store_si128((__m128i *) &l0[idx0 & MASK], _mm_xor_si128(bx0, cx0));
        _mm_store_si128((__m128i *) &l1[idx1 & MASK], _mm_xor_si128(bx1, cx1));
        _mm_store_si128((__m128i *) &l2[idx2 & MASK], _mm_xor_si128(bx2, cx2));

        idx0 = EXTRACT64(cx0);
        idx1 = EXTRACT64(cx1);
        idx2 = EXTRACT64(cx2);

        bx0 = cx0;
        bx1 = cx1;
        bx2 = cx2;

        uint64_t hi, lo, cl, ch;
        cl = ((uint64_t*) &l0[idx0 & MASK])[0];
        ch = ((uint64_t*) &l0[idx0 & MASK])[1];
        lo = __umul128(idx0, cl, &hi);

        al0 += hi;
        ah0 += lo;

        ((uint64_t*) &l0[idx0 & MASK])[0] = al0;
        ((uint64_t*) &l0[idx0 & MASK])[1] = ah0;

        ah0 ^= ch;
        al0 ^= cl;
        idx0 = al0;

        cl = ((uint64_t*) &l1[idx1 & MASK])[0];
        ch = ((uint64_t*) &l1[idx1 & MASK])[1];
        lo = __umul128(idx1, cl, &hi);

        al1 += hi;
        ah1 += lo;

        ((uint64_t*) &l1[idx1 & MASK])[0] = al1;
        ((uint64_t*) &l1[idx1 & MASK])[1] = ah1;

        ah1 ^= ch;
        al1 ^= cl;
        idx1 = al1;

        cl = ((uint64_t*) &l2[idx2 & MASK])[0];
        ch = ((uint64_t*) &l2[idx2 & MASK])[1];
        lo = __umul128(idx2, cl, &hi);

        al2 += hi;
        ah2 += lo;

        ((uint64_t*) &l2[idx2 & MASK])[0] = al2;
        ((uint64_t*) &l2[idx2 & MASK])[1] = ah2;

        ah2 ^= ch;
        al2 ^= cl;
        idx2 = al2;
    }

    cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l0, (__m128i*) h0);
    cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l1, (__m128i*) h1);
    cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) l2, (__m128i*) h2);

    keccakf(h0, 24);
    keccakf(h1, 24);
    keccakf(h2, 24);

    extra_hashes[ctx->state0[0] & 3](ctx->state0, 200, static_cast<char*>(output));
    extra_hashes[ctx->state1[0] & 3](ctx->state1, 200, static_cast<char*>(output) + 32);
    extra_hashes[ctx->state2[0] & 3](ctx->state2, 200, static_cast<char*>(output) + 64);
}

#define CN_STEP1(a, b, c, l, ptr)              \
    a = _mm_xor_si128(a, c);                \
    ptr = (__m128i *)&l[a[0] & MASK];            \
    _mm_prefetch((const char*)ptr, _MM_HINT_T0)

#define CN_STEP2(a, b, c, l, ptr)              \
    c = _mm_load_si128(ptr);              \
    if(SOFT_AES)                        \
        c = soft_aesenc(c, a);              \
    else                            \
        c = _mm_aesenc_si128(c, a);         \
    b = _mm_xor_si128(b, c);                \
    _mm_store_si128(ptr, b); \
    ptr = (__m128i *)&l[c[0] & MASK];            \
    _mm_prefetch((const char*)ptr, _MM_HINT_T0)

#define CN_STEP3(a, b, c, l, ptr)              \
    ;

#define CN_STEP4(a, b, c, l, ptr)              \
    b = _mm_load_si128(ptr);              \
    lo = __umul128(c[0], b[0], &hi);      \
    a = _mm_add_epi64(a, _mm_set_epi64x(lo, hi));       \
    _mm_store_si128(ptr, a)

#define CN_INIT_AND_EXPLODE(lx, hx, i, memory, state)                     \
    const uint8_t* lx = memory + (MEM * i);                                \
    uint64_t* hx = reinterpret_cast<uint64_t*>(state);              \
    keccak((const uint8_t *)input + size * i, size, state, 200);    \
    cn_explode_scratchpad<MEM, SOFT_AES>((__m128i*) hx, (__m128i*) lx)

#define CN_INIT_VARS(ax, bx, cx, hx)                    \
    __m128i ax = _mm_set_epi64x(hx[1] ^ hx[5], hx[0] ^ hx[4]);  \
    __m128i bx = _mm_set_epi64x(hx[3] ^ hx[7], hx[2] ^ hx[6]);  \
    __m128i cx = _mm_set_epi64x(0, 0)

#define CN_IMPLODE_AND_EXPORT(lx, hx, i, output, state)                \
    cn_implode_scratchpad<MEM, SOFT_AES>((__m128i*) lx, (__m128i*) hx); \
    keccakf(hx, 24);                                                    \
    extra_hashes[state[0] & 3](state, 200, static_cast<char*>(output) + (32 * i))


template<size_t ITERATIONS, size_t MEM, size_t MASK, bool SOFT_AES>
inline void cryptonight_penta_hash(const void *__restrict__ input, size_t size, void *__restrict__ output, struct cryptonight_ctx *__restrict__ ctx)
{
    CN_INIT_AND_EXPLODE(l0, h0, 0, ctx->memory, ctx->state0);
    CN_INIT_AND_EXPLODE(l1, h1, 1, ctx->memory, ctx->state1);
    CN_INIT_AND_EXPLODE(l2, h2, 2, ctx->memory, ctx->state2);
    CN_INIT_AND_EXPLODE(l3, h3, 3, ctx->memory, ctx->state3);
    CN_INIT_AND_EXPLODE(l4, h4, 4, ctx->memory, ctx->state4);

    CN_INIT_VARS(ax0, bx0, cx0, h0);
    CN_INIT_VARS(ax1, bx1, cx1, h1);
    CN_INIT_VARS(ax2, bx2, cx2, h2);
    CN_INIT_VARS(ax3, bx3, cx3, h3);
    CN_INIT_VARS(ax4, bx4, cx4, h4);

    for (size_t i = 0; i < ITERATIONS/2; i++) {
        uint64_t hi, lo;
        __m128i *ptr0, *ptr1, *ptr2, *ptr3, *ptr4;

        // EVEN ROUND
        CN_STEP1(ax0, bx0, cx0, l0, ptr0);
        CN_STEP1(ax1, bx1, cx1, l1, ptr1);
        CN_STEP1(ax2, bx2, cx2, l2, ptr2);
        CN_STEP1(ax3, bx3, cx3, l3, ptr3);
        CN_STEP1(ax4, bx4, cx4, l4, ptr4);

        CN_STEP2(ax0, bx0, cx0, l0, ptr0);
        CN_STEP2(ax1, bx1, cx1, l1, ptr1);
        CN_STEP2(ax2, bx2, cx2, l2, ptr2);
        CN_STEP2(ax3, bx3, cx3, l3, ptr3);
        CN_STEP2(ax4, bx4, cx4, l4, ptr4);

        CN_STEP3(ax0, bx0, cx0, l0, ptr0);
        CN_STEP3(ax1, bx1, cx1, l1, ptr1);
        CN_STEP3(ax2, bx2, cx2, l2, ptr2);
        CN_STEP3(ax3, bx3, cx3, l3, ptr3);
        CN_STEP3(ax4, bx4, cx4, l4, ptr4);

        CN_STEP4(ax0, bx0, cx0, l0, ptr0);
        CN_STEP4(ax1, bx1, cx1, l1, ptr1);
        CN_STEP4(ax2, bx2, cx2, l2, ptr2);
        CN_STEP4(ax3, bx3, cx3, l3, ptr3);
        CN_STEP4(ax4, bx4, cx4, l4, ptr4);

        // ODD ROUND
        CN_STEP1(ax0, cx0, bx0, l0, ptr0);
        CN_STEP1(ax1, cx1, bx1, l1, ptr1);
        CN_STEP1(ax2, cx2, bx2, l2, ptr2);
        CN_STEP1(ax3, cx3, bx3, l3, ptr3);
        CN_STEP1(ax4, cx4, bx4, l4, ptr4);

        CN_STEP2(ax0, cx0, bx0, l0, ptr0);
        CN_STEP2(ax1, cx1, bx1, l1, ptr1);
        CN_STEP2(ax2, cx2, bx2, l2, ptr2);
        CN_STEP2(ax3, cx3, bx3, l3, ptr3);
        CN_STEP2(ax4, cx4, bx4, l4, ptr4);

        CN_STEP3(ax0, cx0, bx0, l0, ptr0);
        CN_STEP3(ax1, cx1, bx1, l1, ptr1);
        CN_STEP3(ax2, cx2, bx2, l2, ptr2);
        CN_STEP3(ax3, cx3, bx3, l3, ptr3);
        CN_STEP3(ax4, cx4, bx4, l4, ptr4);

        CN_STEP4(ax0, cx0, bx0, l0, ptr0);
        CN_STEP4(ax1, cx1, bx1, l1, ptr1);
        CN_STEP4(ax2, cx2, bx2, l2, ptr2);
        CN_STEP4(ax3, cx3, bx3, l3, ptr3);
        CN_STEP4(ax4, cx4, bx4, l4, ptr4);
    }

    CN_IMPLODE_AND_EXPORT(l0, h0, 0, output, ctx->state0);
    CN_IMPLODE_AND_EXPORT(l1, h1, 1, output, ctx->state1);
    CN_IMPLODE_AND_EXPORT(l2, h2, 2, output, ctx->state2);
    CN_IMPLODE_AND_EXPORT(l3, h3, 3, output, ctx->state3);
    CN_IMPLODE_AND_EXPORT(l4, h4, 4, output, ctx->state4);
}

#endif /* __CRYPTONIGHT_X86_H__ */