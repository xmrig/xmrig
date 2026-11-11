/*
 * Copyright (c) 2018-2019, tevador <tevador@gmail.com>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 * Copyright (c) 2026, RESBI       <https://github.com/RESBI>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//
// Local header for the LoongArch LASX BLAKE2b build
//   adapted from src/crypto/randomx/blake2/avx2/blake2.h
//
#ifndef BLAKE2_LASX_BLAKE2_H
#define BLAKE2_LASX_BLAKE2_H

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
