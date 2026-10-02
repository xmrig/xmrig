//
// Local header for the LoongArch LASX BLAKE2b build
//   adapted from src/crypto/randomx/blake2/avx2/blake2b.h
//
#ifndef BLAKE2_LASX_BLAKE2B_H
#define BLAKE2_LASX_BLAKE2B_H

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

int blake2b_lasx(void* out, size_t outlen, const void* in, size_t inlen);

#if defined(__cplusplus)
}
#endif

#endif
