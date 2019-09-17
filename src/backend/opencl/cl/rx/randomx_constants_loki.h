/*
Copyright (c) 2019 SChernykh

This file is part of RandomX OpenCL.

RandomX OpenCL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX OpenCL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX OpenCL. If not, see <http://www.gnu.org/licenses/>.
*/

//Dataset base size in bytes. Must be a power of 2.
#define RANDOMX_DATASET_BASE_SIZE  2147483648

//Dataset extra size. Must be divisible by 64.
#define RANDOMX_DATASET_EXTRA_SIZE 33554368

//Scratchpad L3 size in bytes. Must be a power of 2.
#define RANDOMX_SCRATCHPAD_L3      2097152

//Scratchpad L2 size in bytes. Must be a power of two and less than or equal to RANDOMX_SCRATCHPAD_L3.
#define RANDOMX_SCRATCHPAD_L2      262144

//Scratchpad L1 size in bytes. Must be a power of two (minimum 64) and less than or equal to RANDOMX_SCRATCHPAD_L2.
#define RANDOMX_SCRATCHPAD_L1      16384

//Jump condition mask size in bits.
#define RANDOMX_JUMP_BITS          8

//Jump condition mask offset in bits. The sum of RANDOMX_JUMP_BITS and RANDOMX_JUMP_OFFSET must not exceed 16.
#define RANDOMX_JUMP_OFFSET        8

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
#define RANDOMX_FREQ_IROR_R         8
#define RANDOMX_FREQ_IROL_R         2
#define RANDOMX_FREQ_ISWAP_R        4

//Floating point instructions
#define RANDOMX_FREQ_FSWAP_R        4
#define RANDOMX_FREQ_FADD_R        16
#define RANDOMX_FREQ_FADD_M         5
#define RANDOMX_FREQ_FSUB_R        16
#define RANDOMX_FREQ_FSUB_M         5
#define RANDOMX_FREQ_FSCAL_R        6
#define RANDOMX_FREQ_FMUL_R        32
#define RANDOMX_FREQ_FDIV_M         4
#define RANDOMX_FREQ_FSQRT_R        6

//Control instructions
#define RANDOMX_FREQ_CBRANCH       16
#define RANDOMX_FREQ_CFROUND        1

//Store instruction
#define RANDOMX_FREQ_ISTORE        16

//No-op instruction
#define RANDOMX_FREQ_NOP            0

#define RANDOMX_DATASET_ITEM_SIZE 64

#define RANDOMX_PROGRAM_SIZE 320

#define HASH_SIZE 64
#define ENTROPY_SIZE (128 + RANDOMX_PROGRAM_SIZE * 8)
#define REGISTERS_SIZE 256
#define IMM_BUF_SIZE (RANDOMX_PROGRAM_SIZE * 4 - REGISTERS_SIZE)
#define IMM_INDEX_COUNT ((IMM_BUF_SIZE / 4) - 2)
#define VM_STATE_SIZE (REGISTERS_SIZE + IMM_BUF_SIZE + RANDOMX_PROGRAM_SIZE * 4)
#define ROUNDING_MODE (RANDOMX_FREQ_CFROUND ? -1 : 0)

// Scratchpad L1/L2/L3 bits
#define LOC_L1 (32 - 14)
#define LOC_L2 (32 - 18)
#define LOC_L3 (32 - 21)
