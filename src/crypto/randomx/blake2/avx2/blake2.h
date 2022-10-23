#ifndef BLAKE2_AVX2_BLAKE2_H
#define BLAKE2_AVX2_BLAKE2_H

#if !defined(__cplusplus) && (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L)
  #if   defined(_MSC_VER)
    #define INLINE __inline
  #elif defined(__GNUC__)
    #define INLINE __inline__
  #else
    #define INLINE
  #endif
#else
  #define INLINE inline
#endif

#if defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif

enum blake2s_constant {
  BLAKE2S_BLOCKBYTES = 64,
  BLAKE2S_OUTBYTES   = 32,
  BLAKE2S_KEYBYTES   = 32,
  BLAKE2S_SALTBYTES  = 8,
  BLAKE2S_PERSONALBYTES = 8
};

enum blake2b_constant {
  BLAKE2B_BLOCKBYTES = 128,
  BLAKE2B_OUTBYTES   = 64,
  BLAKE2B_KEYBYTES   = 64,
  BLAKE2B_SALTBYTES  = 16,
  BLAKE2B_PERSONALBYTES = 16
};

#endif
