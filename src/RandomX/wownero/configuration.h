/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>
Copyright (c) 2019, Wownero Inc., a Monero Enterprise Alliance partner company

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

#pragma once

//Cache size in KiB. Must be a power of 2.
#define RANDOMX_ARGON_MEMORY       262144

//Number of Argon2d iterations for Cache initialization.
#define RANDOMX_ARGON_ITERATIONS   3

//Number of parallel lanes for Cache initialization.
#define RANDOMX_ARGON_LANES        1

//Argon2d salt
#define RANDOMX_ARGON_SALT         "RandomWOW\x01"

//Number of random Cache accesses per Dataset item. Minimum is 2.
#define RANDOMX_CACHE_ACCESSES     8

//Target latency for SuperscalarHash (in cycles of the reference CPU).
#define RANDOMX_SUPERSCALAR_LATENCY   170

//Dataset base size in bytes. Must be a power of 2.
#define RANDOMX_DATASET_BASE_SIZE  2147483648

//Dataset extra size. Must be divisible by 64.
#define RANDOMX_DATASET_EXTRA_SIZE 33554368

//Number of instructions in a RandomX program. Must be divisible by 8.
#define RANDOMX_PROGRAM_SIZE       256

//Number of iterations during VM execution.
#define RANDOMX_PROGRAM_ITERATIONS 1024

//Number of chained VM executions per hash.
#define RANDOMX_PROGRAM_COUNT      16

//Scratchpad L3 size in bytes. Must be a power of 2.
#define RANDOMX_SCRATCHPAD_L3      1048576

//Scratchpad L2 size in bytes. Must be a power of two and less than or equal to RANDOMX_SCRATCHPAD_L3.
#define RANDOMX_SCRATCHPAD_L2      131072

//Scratchpad L1 size in bytes. Must be a power of two (minimum 64) and less than or equal to RANDOMX_SCRATCHPAD_L2.
#define RANDOMX_SCRATCHPAD_L1      16384

//Jump condition mask size in bits.
#define RANDOMX_JUMP_BITS          8

//Jump condition mask offset in bits. The sum of RANDOMX_JUMP_BITS and RANDOMX_JUMP_OFFSET must not exceed 16.
#define RANDOMX_JUMP_OFFSET        8

/*
Instruction frequencies (per 256 opcodes)
Total sum of frequencies must be 256
*/

//Integer instructions
#define RANDOMX_FREQ_IADD_RS       25
#define RANDOMX_FREQ_IADD_M         7
#define RANDOMX_FREQ_ISUB_R        16
#define RANDOMX_FREQ_ISUB_M         7
#define RANDOMX_FREQ_IMUL_R        16
#define RANDOMX_FREQ_IMUL_M         4
#define RANDOMX_FREQ_IMULH_R        4
#define RANDOMX_FREQ_IMULH_M        1
#define RANDOMX_FREQ_ISMULH_R       4
#define RANDOMX_FREQ_ISMULH_M       1
#define RANDOMX_FREQ_IMUL_RCP       8
#define RANDOMX_FREQ_INEG_R         2
#define RANDOMX_FREQ_IXOR_R        15
#define RANDOMX_FREQ_IXOR_M         5
#define RANDOMX_FREQ_IROR_R        10
#define RANDOMX_FREQ_IROL_R         0
#define RANDOMX_FREQ_ISWAP_R        4

//Floating point instructions
#define RANDOMX_FREQ_FSWAP_R        8
#define RANDOMX_FREQ_FADD_R        20
#define RANDOMX_FREQ_FADD_M         5
#define RANDOMX_FREQ_FSUB_R        20
#define RANDOMX_FREQ_FSUB_M         5
#define RANDOMX_FREQ_FSCAL_R        6
#define RANDOMX_FREQ_FMUL_R        20
#define RANDOMX_FREQ_FDIV_M         4
#define RANDOMX_FREQ_FSQRT_R        6

//Control instructions
#define RANDOMX_FREQ_CBRANCH       16
#define RANDOMX_FREQ_CFROUND        1

//Store instruction
#define RANDOMX_FREQ_ISTORE        16

//No-op instruction
#define RANDOMX_FREQ_NOP            0
/*                               ------
                                  256
*/
