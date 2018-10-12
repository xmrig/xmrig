/*
 * Copyright 2008  Veselin Georgiev,
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

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "libcpuid.h"
#include "libcpuid_util.h"
#include "libcpuid_internal.h"
#include "recog_amd.h"

const struct amd_code_str { amd_code_t code; char *str; } amd_code_str[] = {
	#define CODE(x) { x, #x }
	#define CODE2(x, y) CODE(x)
	#include "amd_code_t.h"
	#undef CODE
};

struct amd_code_and_bits_t {
	int code;
	uint64_t bits;
};

enum _amd_model_codes_t {
	// Only for Ryzen CPUs:
	_1400,
	_1500,
	_1600,
	_1900,
	_2400,
	_2500,
	_2700,
};

static void load_amd_features(struct cpu_raw_data_t* raw, struct cpu_id_t* data)
{
	const struct feature_map_t matchtable_edx81[] = {
		{ 20, CPU_FEATURE_NX },
		{ 22, CPU_FEATURE_MMXEXT },
		{ 25, CPU_FEATURE_FXSR_OPT },
		{ 30, CPU_FEATURE_3DNOWEXT },
		{ 31, CPU_FEATURE_3DNOW },
	};
	const struct feature_map_t matchtable_ecx81[] = {
		{  1, CPU_FEATURE_CMP_LEGACY },
		{  2, CPU_FEATURE_SVM },
		{  5, CPU_FEATURE_ABM },
		{  6, CPU_FEATURE_SSE4A },
		{  7, CPU_FEATURE_MISALIGNSSE },
		{  8, CPU_FEATURE_3DNOWPREFETCH },
		{  9, CPU_FEATURE_OSVW },
		{ 10, CPU_FEATURE_IBS },
		{ 11, CPU_FEATURE_XOP },
		{ 12, CPU_FEATURE_SKINIT },
		{ 13, CPU_FEATURE_WDT },
		{ 16, CPU_FEATURE_FMA4 },
		{ 21, CPU_FEATURE_TBM },
	};
	const struct feature_map_t matchtable_edx87[] = {
		{  0, CPU_FEATURE_TS },
		{  1, CPU_FEATURE_FID },
		{  2, CPU_FEATURE_VID },
		{  3, CPU_FEATURE_TTP },
		{  4, CPU_FEATURE_TM_AMD },
		{  5, CPU_FEATURE_STC },
		{  6, CPU_FEATURE_100MHZSTEPS },
		{  7, CPU_FEATURE_HWPSTATE },
		/* id 8 is handled in common */
		{  9, CPU_FEATURE_CPB },
		{ 10, CPU_FEATURE_APERFMPERF },
		{ 11, CPU_FEATURE_PFI },
		{ 12, CPU_FEATURE_PA },
	};
	if (raw->ext_cpuid[0][0] >= 0x80000001) {
		match_features(matchtable_edx81, COUNT_OF(matchtable_edx81), raw->ext_cpuid[1][3], data);
		match_features(matchtable_ecx81, COUNT_OF(matchtable_ecx81), raw->ext_cpuid[1][2], data);
	}
	if (raw->ext_cpuid[0][0] >= 0x80000007)
		match_features(matchtable_edx87, COUNT_OF(matchtable_edx87), raw->ext_cpuid[7][3], data);
	if (raw->ext_cpuid[0][0] >= 0x8000001a) {
		/* We have the extended info about SSE unit size */
		data->detection_hints[CPU_HINT_SSE_SIZE_AUTH] = 1;
		data->sse_size = (raw->ext_cpuid[0x1a][0] & 1) ? 128 : 64;
	}
}

static void decode_amd_cache_info(struct cpu_raw_data_t* raw, struct cpu_id_t* data)
{
	int l3_result;
	const int assoc_table[16] = {
		0, 1, 2, 0, 4, 0, 8, 0, 16, 0, 32, 48, 64, 96, 128, 255
	};
	unsigned n = raw->ext_cpuid[0][0];
	
	if (n >= 0x80000005) {
		data->l1_data_cache = (raw->ext_cpuid[5][2] >> 24) & 0xff;
		data->l1_assoc = (raw->ext_cpuid[5][2] >> 16) & 0xff;
		data->l1_cacheline = (raw->ext_cpuid[5][2]) & 0xff;
		data->l1_instruction_cache = (raw->ext_cpuid[5][3] >> 24) & 0xff;
	}
	if (n >= 0x80000006) {
		data->l2_cache = (raw->ext_cpuid[6][2] >> 16) & 0xffff;
		data->l2_assoc = assoc_table[(raw->ext_cpuid[6][2] >> 12) & 0xf];
		data->l2_cacheline = (raw->ext_cpuid[6][2]) & 0xff;
		
		l3_result = (raw->ext_cpuid[6][3] >> 18);
		if (l3_result > 0) {
			l3_result = 512 * l3_result; /* AMD spec says it's a range,
			                                but we take the lower bound */
			data->l3_cache = l3_result;
			data->l3_assoc = assoc_table[(raw->ext_cpuid[6][3] >> 12) & 0xf];
			data->l3_cacheline = (raw->ext_cpuid[6][3]) & 0xff;
		} else {
			data->l3_cache = 0;
		}
	}
}

static void decode_amd_number_of_cores(struct cpu_raw_data_t* raw, struct cpu_id_t* data)
{
	int logical_cpus = -1, num_cores = -1;
	
	if (raw->basic_cpuid[0][0] >= 1) {
		logical_cpus = (raw->basic_cpuid[1][1] >> 16) & 0xff;
		if (raw->ext_cpuid[0][0] >= 8) {
			num_cores = 1 + (raw->ext_cpuid[8][2] & 0xff);
		}
	}
	if (data->flags[CPU_FEATURE_HT]) {
		if (num_cores > 1) {
			if (data->ext_family >= 23)
				num_cores /= 2; // e.g., Ryzen 7 reports 16 "real" cores, but they are really just 8.
			data->num_cores = num_cores;
			data->num_logical_cpus = logical_cpus;
		} else {
			data->num_cores = 1;
			data->num_logical_cpus = (logical_cpus >= 2 ? logical_cpus : 2);
		}
	} else {
		data->num_cores = data->num_logical_cpus = 1;
	}
}

int cpuid_identify_amd(struct cpu_raw_data_t* raw, struct cpu_id_t* data, struct internal_id_info_t* internal)
{
	load_amd_features(raw, data);
	decode_amd_cache_info(raw, data);
	decode_amd_number_of_cores(raw, data);
	return 0;
}
