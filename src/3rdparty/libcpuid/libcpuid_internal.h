/*
 * Copyright 2016  Veselin Georgiev,
 * anrieffNOSPAM @ mgail_DOT.com (convert to gmail)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef __LIBCPUID_INTERNAL_H__
#define __LIBCPUID_INTERNAL_H__
/*
 * This file contains internal undocumented declarations and function prototypes
 * for the workings of the internal library infrastructure.
 */

enum _common_codes_t {
	NA = 0,
	NC, /* No code */
};

#define CODE(x) x
#define CODE2(x, y) x = y
enum _amd_code_t {
	#include "amd_code_t.h"
};
typedef enum _amd_code_t amd_code_t;

enum _intel_code_t {
	#include "intel_code_t.h"
};
typedef enum _intel_code_t intel_code_t;
#undef CODE
#undef CODE2

struct internal_id_info_t {
	union {
		amd_code_t   amd;
		intel_code_t intel;
	} code;
	uint64_t bits;
	int score; // detection (matchtable) score
};

#define LBIT(x) (((long long) 1) << x)

enum _common_bits_t {
	_M_                     = LBIT(  0 ),
	MOBILE_                 = LBIT(  1 ),
	_MP_                    = LBIT(  2 ),
};

// additional detection bits for Intel CPUs:
enum _intel_bits_t {
	PENTIUM_                = LBIT( 10 ),
	CELERON_                = LBIT( 11 ),
	CORE_                   = LBIT( 12 ),
	_I_                     = LBIT( 13 ),
	_3                      = LBIT( 14 ),
	_5                      = LBIT( 15 ),
	_7                      = LBIT( 16 ),
	_9                      = LBIT( 17 ),
	XEON_                   = LBIT( 18 ),
	ATOM_                   = LBIT( 19 ),
};
typedef enum _intel_bits_t intel_bits_t;

enum _amd_bits_t {
	ATHLON_      = LBIT( 10 ),
	_XP_         = LBIT( 11 ),
	DURON_       = LBIT( 12 ),
	SEMPRON_     = LBIT( 13 ),
	OPTERON_     = LBIT( 14 ),
	TURION_      = LBIT( 15 ),
	_LV_         = LBIT( 16 ),
	_64_         = LBIT( 17 ),
	_X2          = LBIT( 18 ),
	_X3          = LBIT( 19 ),
	_X4          = LBIT( 20 ),
	_X6          = LBIT( 21 ),
	_FX          = LBIT( 22 ),
	_APU_        = LBIT( 23 ),
};
typedef enum _amd_bits_t amd_bits_t;



int cpu_ident_internal(struct cpu_raw_data_t* raw, struct cpu_id_t* data, 
		       struct internal_id_info_t* internal);

#endif /* __LIBCPUID_INTERNAL_H__ */
