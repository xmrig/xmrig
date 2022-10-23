#ifndef BLAKE2_AVX2_BLAKE2B_H
#define BLAKE2_AVX2_BLAKE2B_H

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

int blake2b_avx2(void* out, size_t outlen, const void* in, size_t inlen);

#if defined(__cplusplus)
}
#endif

#endif
