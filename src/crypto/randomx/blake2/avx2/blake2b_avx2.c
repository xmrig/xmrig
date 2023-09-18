#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "blake2.h"
#include "blake2b.h"
#include "blake2b-common.h"

ALIGN(64) static const uint64_t blake2b_IV[8] = {
  UINT64_C(0x6A09E667F3BCC908), UINT64_C(0xBB67AE8584CAA73B),
  UINT64_C(0x3C6EF372FE94F82B), UINT64_C(0xA54FF53A5F1D36F1),
  UINT64_C(0x510E527FADE682D1), UINT64_C(0x9B05688C2B3E6C1F),
  UINT64_C(0x1F83D9ABFB41BD6B), UINT64_C(0x5BE0CD19137E2179),
};

#define BLAKE2B_G1_V1(a, b, c, d, m) do {     \
  a = ADD(a, m);                              \
  a = ADD(a, b); d = XOR(d, a); d = ROT32(d); \
  c = ADD(c, d); b = XOR(b, c); b = ROT24(b); \
} while(0)

#define BLAKE2B_G2_V1(a, b, c, d, m) do {     \
  a = ADD(a, m);                              \
  a = ADD(a, b); d = XOR(d, a); d = ROT16(d); \
  c = ADD(c, d); b = XOR(b, c); b = ROT63(b); \
} while(0)

#define BLAKE2B_DIAG_V1(a, b, c, d) do {                 \
  a = _mm256_permute4x64_epi64(a, _MM_SHUFFLE(2,1,0,3)); \
  d = _mm256_permute4x64_epi64(d, _MM_SHUFFLE(1,0,3,2)); \
  c = _mm256_permute4x64_epi64(c, _MM_SHUFFLE(0,3,2,1)); \
} while(0)

#define BLAKE2B_UNDIAG_V1(a, b, c, d) do {               \
  a = _mm256_permute4x64_epi64(a, _MM_SHUFFLE(0,3,2,1)); \
  d = _mm256_permute4x64_epi64(d, _MM_SHUFFLE(1,0,3,2)); \
  c = _mm256_permute4x64_epi64(c, _MM_SHUFFLE(2,1,0,3)); \
} while(0)

#include "blake2b-load-avx2.h"

#define BLAKE2B_ROUND_V1(a, b, c, d, r, m) do { \
  __m256i b0;                                   \
  BLAKE2B_LOAD_MSG_ ##r ##_1(b0);               \
  BLAKE2B_G1_V1(a, b, c, d, b0);                \
  BLAKE2B_LOAD_MSG_ ##r ##_2(b0);               \
  BLAKE2B_G2_V1(a, b, c, d, b0);                \
  BLAKE2B_DIAG_V1(a, b, c, d);                  \
  BLAKE2B_LOAD_MSG_ ##r ##_3(b0);               \
  BLAKE2B_G1_V1(a, b, c, d, b0);                \
  BLAKE2B_LOAD_MSG_ ##r ##_4(b0);               \
  BLAKE2B_G2_V1(a, b, c, d, b0);                \
  BLAKE2B_UNDIAG_V1(a, b, c, d);                \
} while(0)

#define BLAKE2B_ROUNDS_V1(a, b, c, d, m) do {   \
  BLAKE2B_ROUND_V1(a, b, c, d,  0, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d,  1, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d,  2, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d,  3, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d,  4, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d,  5, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d,  6, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d,  7, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d,  8, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d,  9, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d, 10, (m));        \
  BLAKE2B_ROUND_V1(a, b, c, d, 11, (m));        \
} while(0)

#define DECLARE_MESSAGE_WORDS(m)                                       \
  const __m256i m0 = _mm256_broadcastsi128_si256(LOADU128((m) +   0)); \
  const __m256i m1 = _mm256_broadcastsi128_si256(LOADU128((m) +  16)); \
  const __m256i m2 = _mm256_broadcastsi128_si256(LOADU128((m) +  32)); \
  const __m256i m3 = _mm256_broadcastsi128_si256(LOADU128((m) +  48)); \
  const __m256i m4 = _mm256_broadcastsi128_si256(LOADU128((m) +  64)); \
  const __m256i m5 = _mm256_broadcastsi128_si256(LOADU128((m) +  80)); \
  const __m256i m6 = _mm256_broadcastsi128_si256(LOADU128((m) +  96)); \
  const __m256i m7 = _mm256_broadcastsi128_si256(LOADU128((m) + 112)); \
  __m256i t0, t1;

#define BLAKE2B_COMPRESS_V1(a, b, m, t0, t1, f0, f1) do { \
  DECLARE_MESSAGE_WORDS(m)                                \
  const __m256i iv0 = a;                                  \
  const __m256i iv1 = b;                                  \
  __m256i c = LOAD(&blake2b_IV[0]);                       \
  __m256i d = XOR(                                        \
    LOAD(&blake2b_IV[4]),                                 \
    _mm256_set_epi64x(f1, f0, t1, t0)                     \
  );                                                      \
  BLAKE2B_ROUNDS_V1(a, b, c, d, m);                       \
  a = XOR(a, c);                                          \
  b = XOR(b, d);                                          \
  a = XOR(a, iv0);                                        \
  b = XOR(b, iv1);                                        \
} while(0)

int blake2b_avx2(void* out_ptr, size_t outlen, const void* in_ptr, size_t inlen) {
  const __m256i parameter_block = _mm256_set_epi64x(0, 0, 0, 0x01010000UL | (uint32_t)outlen);
  ALIGN(64) uint8_t buffer[BLAKE2B_BLOCKBYTES];
  __m256i a = XOR(LOAD(&blake2b_IV[0]), parameter_block);
  __m256i b = LOAD(&blake2b_IV[4]);
  uint64_t counter = 0;
  const uint8_t* in = (const uint8_t*)in_ptr;
  do {
    const uint64_t flag = (inlen <= BLAKE2B_BLOCKBYTES) ? -1 : 0;
    size_t block_size = BLAKE2B_BLOCKBYTES;
    if(inlen < BLAKE2B_BLOCKBYTES) {
      memcpy(buffer, in, inlen);
      memset(buffer + inlen, 0, BLAKE2B_BLOCKBYTES - inlen);
      block_size = inlen;
      in = buffer;
    }
    counter += block_size;
    BLAKE2B_COMPRESS_V1(a, b, in, counter, 0, flag, 0);
    inlen -= block_size;
    in    += block_size;
  } while(inlen > 0);

  uint8_t* out = (uint8_t*)out_ptr;

  switch (outlen) {
  case 64:
      STOREU(out + 32, b);
      // Fall through

  case 32:
      STOREU(out, a);
      break;

  default:
      STOREU(buffer, a);
      STOREU(buffer + 32, b);
      memcpy(out, buffer, outlen);
      break;
  }

  _mm256_zeroupper();
  return 0;
}
