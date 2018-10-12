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
#include <string.h>
#include <ctype.h>
#include "libcpuid.h"
#include "libcpuid_util.h"
#include "libcpuid_internal.h"
#include "recog_intel.h"

const struct intel_bcode_str { intel_code_t code; char *str; } intel_bcode_str[] = {
	#define CODE(x) { x, #x }
	#define CODE2(x, y) CODE(x)
	#include "intel_code_t.h"
	#undef CODE
};

typedef struct {
	int code;
	uint64_t bits;
} intel_code_and_bits_t;

enum _intel_model_t {
	UNKNOWN = -1,
	_3000 = 100,
	_3100,
	_3200,
	X3200,
	_3300,
	X3300,
	_5100,
	_5200,
	_5300,
	_5400,
	_2xxx, /* Core i[357] 2xxx */
	_3xxx, /* Core i[357] 3xxx */
};
typedef enum _intel_model_t intel_model_t;

static void load_intel_features(struct cpu_raw_data_t* raw, struct cpu_id_t* data)
{
	const struct feature_map_t matchtable_edx1[] = {
		{ 18, CPU_FEATURE_PN },
		{ 21, CPU_FEATURE_DTS },
		{ 22, CPU_FEATURE_ACPI },
		{ 27, CPU_FEATURE_SS },
		{ 29, CPU_FEATURE_TM },
		{ 30, CPU_FEATURE_IA64 },
		{ 31, CPU_FEATURE_PBE },
	};
	const struct feature_map_t matchtable_ecx1[] = {
		{  2, CPU_FEATURE_DTS64 },
		{  4, CPU_FEATURE_DS_CPL },
		{  5, CPU_FEATURE_VMX },
		{  6, CPU_FEATURE_SMX },
		{  7, CPU_FEATURE_EST },
		{  8, CPU_FEATURE_TM2 },
		{ 10, CPU_FEATURE_CID },
		{ 14, CPU_FEATURE_XTPR },
		{ 15, CPU_FEATURE_PDCM },
		{ 18, CPU_FEATURE_DCA },
		{ 21, CPU_FEATURE_X2APIC },
	};
	const struct feature_map_t matchtable_edx81[] = {
		{ 20, CPU_FEATURE_XD },
	};
	const struct feature_map_t matchtable_ebx7[] = {
		{  2, CPU_FEATURE_SGX },
		{  4, CPU_FEATURE_HLE },
		{ 11, CPU_FEATURE_RTM },
		{ 16, CPU_FEATURE_AVX512F },
		{ 17, CPU_FEATURE_AVX512DQ },
		{ 18, CPU_FEATURE_RDSEED },
		{ 19, CPU_FEATURE_ADX },
		{ 26, CPU_FEATURE_AVX512PF },
		{ 27, CPU_FEATURE_AVX512ER },
		{ 28, CPU_FEATURE_AVX512CD },
		{ 29, CPU_FEATURE_SHA_NI },
		{ 30, CPU_FEATURE_AVX512BW },
		{ 31, CPU_FEATURE_AVX512VL },
	};
	if (raw->basic_cpuid[0][0] >= 1) {
		match_features(matchtable_edx1, COUNT_OF(matchtable_edx1), raw->basic_cpuid[1][3], data);
		match_features(matchtable_ecx1, COUNT_OF(matchtable_ecx1), raw->basic_cpuid[1][2], data);
	}
	if (raw->ext_cpuid[0][0] >= 1) {
		match_features(matchtable_edx81, COUNT_OF(matchtable_edx81), raw->ext_cpuid[1][3], data);
	}
	// detect TSX/AVX512:
	if (raw->basic_cpuid[0][0] >= 7) {
		match_features(matchtable_ebx7, COUNT_OF(matchtable_ebx7), raw->basic_cpuid[7][1], data);
	}
}

enum _cache_type_t {
	L1I,
	L1D,
	L2,
	L3,
	L4
};
typedef enum _cache_type_t cache_type_t;

static void check_case(uint8_t on, cache_type_t cache, int size, int assoc, int linesize, struct cpu_id_t* data)
{
	if (!on) return;
	switch (cache) {
		case L1I:
			data->l1_instruction_cache = size;
			break;
		case L1D:
			data->l1_data_cache = size;
			data->l1_assoc = assoc;
			data->l1_cacheline = linesize;
			break;
		case L2:
			data->l2_cache = size;
			data->l2_assoc = assoc;
			data->l2_cacheline = linesize;
			break;
		case L3:
			data->l3_cache = size;
			data->l3_assoc = assoc;
			data->l3_cacheline = linesize;
			break;
		case L4:
			data->l4_cache = size;
			data->l4_assoc = assoc;
			data->l4_cacheline = linesize;
			break;
		default:
			break;
	}
}

static void decode_intel_oldstyle_cache_info(struct cpu_raw_data_t* raw, struct cpu_id_t* data)
{
	uint8_t f[256] = {0};
	int reg, off;
	uint32_t x;
	for (reg = 0; reg < 4; reg++) {
		x = raw->basic_cpuid[2][reg];
		if (x & 0x80000000) continue;
		for (off = 0; off < 4; off++) {
			f[x & 0xff] = 1;
			x >>= 8;
		}
	}
	
	check_case(f[0x06], L1I,      8,  4,  32, data);
	check_case(f[0x08], L1I,     16,  4,  32, data);
	check_case(f[0x0A], L1D,      8,  2,  32, data);
	check_case(f[0x0C], L1D,     16,  4,  32, data);
	check_case(f[0x22],  L3,    512,  4,  64, data);
	check_case(f[0x23],  L3,   1024,  8,  64, data);
	check_case(f[0x25],  L3,   2048,  8,  64, data);
	check_case(f[0x29],  L3,   4096,  8,  64, data);
	check_case(f[0x2C], L1D,     32,  8,  64, data);
	check_case(f[0x30], L1I,     32,  8,  64, data);
	check_case(f[0x39],  L2,    128,  4,  64, data);
	check_case(f[0x3A],  L2,    192,  6,  64, data);
	check_case(f[0x3B],  L2,    128,  2,  64, data);
	check_case(f[0x3C],  L2,    256,  4,  64, data);
	check_case(f[0x3D],  L2,    384,  6,  64, data);
	check_case(f[0x3E],  L2,    512,  4,  64, data);
	check_case(f[0x41],  L2,    128,  4,  32, data);
	check_case(f[0x42],  L2,    256,  4,  32, data);
	check_case(f[0x43],  L2,    512,  4,  32, data);
	check_case(f[0x44],  L2,   1024,  4,  32, data);
	check_case(f[0x45],  L2,   2048,  4,  32, data);
	check_case(f[0x46],  L3,   4096,  4,  64, data);
	check_case(f[0x47],  L3,   8192,  8,  64, data);
	check_case(f[0x4A],  L3,   6144, 12,  64, data);
	check_case(f[0x4B],  L3,   8192, 16,  64, data);
	check_case(f[0x4C],  L3,  12288, 12,  64, data);
	check_case(f[0x4D],  L3,  16384, 16,  64, data);
	check_case(f[0x4E],  L2,   6144, 24,  64, data);
	check_case(f[0x60], L1D,     16,  8,  64, data);
	check_case(f[0x66], L1D,      8,  4,  64, data);
	check_case(f[0x67], L1D,     16,  4,  64, data);
	check_case(f[0x68], L1D,     32,  4,  64, data);
	/* The following four entries are trace cache. Intel does not
	 * specify a cache-line size, so we use -1 instead
	 */
	check_case(f[0x70], L1I,     12,  8,  -1, data);
	check_case(f[0x71], L1I,     16,  8,  -1, data);
	check_case(f[0x72], L1I,     32,  8,  -1, data);
	check_case(f[0x73], L1I,     64,  8,  -1, data);
	
	check_case(f[0x78],  L2,   1024,  4,  64, data);
	check_case(f[0x79],  L2,    128,  8,  64, data);
	check_case(f[0x7A],  L2,    256,  8,  64, data);
	check_case(f[0x7B],  L2,    512,  8,  64, data);
	check_case(f[0x7C],  L2,   1024,  8,  64, data);
	check_case(f[0x7D],  L2,   2048,  8,  64, data);
	check_case(f[0x7F],  L2,    512,  2,  64, data);
	check_case(f[0x82],  L2,    256,  8,  32, data);
	check_case(f[0x83],  L2,    512,  8,  32, data);
	check_case(f[0x84],  L2,   1024,  8,  32, data);
	check_case(f[0x85],  L2,   2048,  8,  32, data);
	check_case(f[0x86],  L2,    512,  4,  64, data);
	check_case(f[0x87],  L2,   1024,  8,  64, data);
	
	if (f[0x49]) {
		/* This flag is overloaded with two meanings. On Xeon MP
		 * (family 0xf, model 0x6) this means L3 cache. On all other
		 * CPUs (notably Conroe et al), this is L2 cache. In both cases
		 * it means 4MB, 16-way associative, 64-byte line size.
		 */
		if (data->family == 0xf && data->model == 0x6) {
			data->l3_cache = 4096;
			data->l3_assoc = 16;
			data->l3_cacheline = 64;
		} else {
			data->l2_cache = 4096;
			data->l2_assoc = 16;
			data->l2_cacheline = 64;
		}
	}
	if (f[0x40]) {
		/* Again, a special flag. It means:
		 * 1) If no L2 is specified, then CPU is w/o L2 (0 KB)
		 * 2) If L2 is specified by other flags, then, CPU is w/o L3.
		 */
		if (data->l2_cache == -1) {
			data->l2_cache = 0;
		} else {
			data->l3_cache = 0;
		}
	}
}

static void decode_intel_deterministic_cache_info(struct cpu_raw_data_t* raw,
                                                  struct cpu_id_t* data)
{
	int ecx;
	int ways, partitions, linesize, sets, size, level, typenumber;
	cache_type_t type;
	for (ecx = 0; ecx < MAX_INTELFN4_LEVEL; ecx++) {
		typenumber = raw->intel_fn4[ecx][0] & 0x1f;
		if (typenumber == 0) break;
		level = (raw->intel_fn4[ecx][0] >> 5) & 0x7;
		if (level == 1 && typenumber == 1)
			type = L1D;
		else if (level == 1 && typenumber == 2)
			type = L1I;
		else if (level == 2 && typenumber == 3)
			type = L2;
		else if (level == 3 && typenumber == 3)
			type = L3;
		else if (level == 4 && typenumber == 3)
			type = L4;
		else {
			continue;
		}
		ways = ((raw->intel_fn4[ecx][1] >> 22) & 0x3ff) + 1;
		partitions = ((raw->intel_fn4[ecx][1] >> 12) & 0x3ff) + 1;
		linesize = (raw->intel_fn4[ecx][1] & 0xfff) + 1;
		sets = raw->intel_fn4[ecx][2] + 1;
		size = ways * partitions * linesize * sets / 1024;
		check_case(1, type, size, ways, linesize, data);
	}
}

static int decode_intel_extended_topology(struct cpu_raw_data_t* raw,
                                           struct cpu_id_t* data)
{
	int i, level_type, num_smt = -1, num_core = -1;
	for (i = 0; i < MAX_INTELFN11_LEVEL; i++) {
		level_type = (raw->intel_fn11[i][2] & 0xff00) >> 8;
		switch (level_type) {
			case 0x01:
				num_smt = raw->intel_fn11[i][1] & 0xffff;
				break;
			case 0x02:
				num_core = raw->intel_fn11[i][1] & 0xffff;
				break;
			default:
				break;
		}
	}
	if (num_smt == -1 || num_core == -1) return 0;
	data->num_logical_cpus = num_core;
	data->num_cores = num_core / num_smt;
	// make sure num_cores is at least 1. In VMs, the CPUID instruction
	// is rigged and may give nonsensical results, but we should at least
	// avoid outputs like data->num_cores == 0.
	if (data->num_cores <= 0) data->num_cores = 1;
	return 1;
}

static void decode_intel_number_of_cores(struct cpu_raw_data_t* raw,
                                         struct cpu_id_t* data)
{
	int logical_cpus = -1, num_cores = -1;
	
	if (raw->basic_cpuid[0][0] >= 11) {
		if (decode_intel_extended_topology(raw, data)) return;
	}
	
	if (raw->basic_cpuid[0][0] >= 1) {
		logical_cpus = (raw->basic_cpuid[1][1] >> 16) & 0xff;
		if (raw->basic_cpuid[0][0] >= 4) {
			num_cores = 1 + ((raw->basic_cpuid[4][0] >> 26) & 0x3f);
		}
	}
	if (data->flags[CPU_FEATURE_HT]) {
		if (num_cores > 1) {
			data->num_cores = num_cores;
			data->num_logical_cpus = logical_cpus;
		} else {
			data->num_cores = 1;
			data->num_logical_cpus = (logical_cpus >= 1 ? logical_cpus : 1);
			if (data->num_logical_cpus == 1)
				data->flags[CPU_FEATURE_HT] = 0;
		}
	} else {
		data->num_cores = data->num_logical_cpus = 1;
	}
}

static intel_code_and_bits_t get_brand_code_and_bits(struct cpu_id_t* data)
{
	intel_code_t code = (intel_code_t) NC;
	intel_code_and_bits_t result;
	uint64_t bits = 0;
	int i = 0;
	const char* bs = data->brand_str;
	const char* s;
	const struct { intel_code_t c; const char *search; } matchtable[] = {
		{ PENTIUM_M, "Pentium(R) M" },
		{ CORE_SOLO, "Pentium(R) Dual  CPU" },
		{ CORE_SOLO, "Pentium(R) Dual-Core" },
		{ PENTIUM_D, "Pentium(R) D" },
		{ CORE_SOLO, "Genuine Intel(R) CPU" },
		{ CORE_SOLO, "Intel(R) Core(TM)" },
		{ DIAMONDVILLE, "CPU [N ][23]## " },
		{ SILVERTHORNE, "CPU Z" },
		{ PINEVIEW, "CPU [ND][45]## " },
		{ CEDARVIEW, "CPU [ND]#### " },
	};
	
	const struct { uint64_t bit; const char* search; } bit_matchtable[] = {
		{ XEON_, "Xeon" },
		{ _MP_, " MP" },
		{ ATOM_, "Atom(TM) CPU" },
		{ MOBILE_, "Mobile" },
		{ CELERON_, "Celeron" },
		{ PENTIUM_, "Pentium" },
	};
	
	for (i = 0; i < COUNT_OF(bit_matchtable); i++) {
		if (match_pattern(bs, bit_matchtable[i].search))
			bits |= bit_matchtable[i].bit;
	}
	
	if ((i = match_pattern(bs, "Core(TM) [im][3579]")) != 0) {
		bits |= CORE_;
		i--;
		switch (bs[i + 9]) {
			case 'i': bits |= _I_; break;
			case 'm': bits |= _M_; break;
		}
		switch (bs[i + 10]) {
			case '3': bits |= _3; break;
			case '5': bits |= _5; break;
			case '7': bits |= _7; break;
			case '9': bits |= _9; break;
		}
	}
	for (i = 0; i < COUNT_OF(matchtable); i++)
		if (match_pattern(bs, matchtable[i].search)) {
			code = matchtable[i].c;
			break;
		}
	if (bits & XEON_) {
		if (match_pattern(bs, "W35##") || match_pattern(bs, "[ELXW]75##"))
			bits |= _7;
		else if (match_pattern(bs, "[ELXW]55##"))
			code = GAINESTOWN;
		else if (match_pattern(bs, "[ELXW]56##"))
			code = WESTMERE;
		else if (data->l3_cache > 0 && data->family == 16)
			/* restrict by family, since later Xeons also have L3 ... */
			code = IRWIN;
	}
	if (match_all(bits, XEON_ + _MP_) && data->l3_cache > 0)
		code = POTOMAC;
	if (code == CORE_SOLO) {
		s = strstr(bs, "CPU");
		if (s) {
			s += 3;
			while (*s == ' ') s++;
			if (*s == 'T')
				bits |= MOBILE_;
		}
	}
	if (code == CORE_SOLO) {
		switch (data->num_cores) {
			case 1: break;
			case 2:
			{
				code = CORE_DUO;
				if (data->num_logical_cpus > 2)
					code = DUAL_CORE_HT;
				break;
			}
			case 4:
			{
				code = QUAD_CORE;
				if (data->num_logical_cpus > 4)
					code = QUAD_CORE_HT;
				break;
			}
			default:
				code = MORE_THAN_QUADCORE; break;
		}
	}
	
	if (code == CORE_DUO && (bits & MOBILE_) && data->model != 14) {
		if (data->ext_model < 23) {
			code = MEROM;
		} else {
			code = PENRYN;
		}
	}
	if (data->ext_model == 23 &&
		(code == CORE_DUO || code == PENTIUM_D || (bits & CELERON_))) {
		code = WOLFDALE;
	}

	result.code = code;
	result.bits = bits;
	return result;
}

static void decode_intel_sgx_features(const struct cpu_raw_data_t* raw, struct cpu_id_t* data)
{
	struct cpu_epc_t epc;
	int i;
	
	if (raw->basic_cpuid[0][0] < 0x12) return; // no 12h leaf
	if (raw->basic_cpuid[0x12][0] == 0) return; // no sub-leafs available, probably it's disabled by BIOS
	
	// decode sub-leaf 0:
	if (raw->basic_cpuid[0x12][0] & 1) data->sgx.flags[INTEL_SGX1] = 1;
	if (raw->basic_cpuid[0x12][0] & 2) data->sgx.flags[INTEL_SGX2] = 1;
	if (data->sgx.flags[INTEL_SGX1] || data->sgx.flags[INTEL_SGX2])
		data->sgx.present = 1;
	data->sgx.misc_select = raw->basic_cpuid[0x12][1];
	data->sgx.max_enclave_32bit = (raw->basic_cpuid[0x12][3]     ) & 0xff;
	data->sgx.max_enclave_64bit = (raw->basic_cpuid[0x12][3] >> 8) & 0xff;
	
	// decode sub-leaf 1:
	data->sgx.secs_attributes = raw->intel_fn12h[1][0] | (((uint64_t) raw->intel_fn12h[1][1]) << 32);
	data->sgx.secs_xfrm       = raw->intel_fn12h[1][2] | (((uint64_t) raw->intel_fn12h[1][3]) << 32);
	
	// decode higher-order subleafs, whenever present:
	data->sgx.num_epc_sections = -1;
	for (i = 0; i < 1000000; i++) {
		epc = cpuid_get_epc(i, raw);
		if (epc.length == 0) {
			data->sgx.num_epc_sections = i;
			break;
		}
	}
	if (data->sgx.num_epc_sections == -1) {
		data->sgx.num_epc_sections = 1000000;
	}
}

struct cpu_epc_t cpuid_get_epc(int index, const struct cpu_raw_data_t* raw)
{
	uint32_t regs[4];
	struct cpu_epc_t retval = {0, 0};
	if (raw && index < MAX_INTELFN12H_LEVEL - 2) {
		// this was queried already, use the data:
		memcpy(regs, raw->intel_fn12h[2 + index], sizeof(regs));
	} else {
		// query this ourselves:
		regs[0] = 0x12;
		regs[2] = 2 + index;
		regs[1] = regs[3] = 0;
		cpu_exec_cpuid_ext(regs);
	}
	
	// decode values:
	if ((regs[0] & 0xf) == 0x1) {
		retval.start_addr |= (regs[0] & 0xfffff000); // bits [12, 32) -> bits [12, 32)
		retval.start_addr |= ((uint64_t) (regs[1] & 0x000fffff)) << 32; // bits [0, 20) -> bits [32, 52)
		retval.length     |= (regs[2] & 0xfffff000); // bits [12, 32) -> bits [12, 32)
		retval.length     |= ((uint64_t) (regs[3] & 0x000fffff)) << 32; // bits [0, 20) -> bits [32, 52)
	}
	return retval;
}

int cpuid_identify_intel(struct cpu_raw_data_t* raw, struct cpu_id_t* data, struct internal_id_info_t* internal)
{
	intel_code_and_bits_t brand;

	load_intel_features(raw, data);
	if (raw->basic_cpuid[0][0] >= 4) {
		/* Deterministic way is preferred, being more generic */
		decode_intel_deterministic_cache_info(raw, data);
	} else if (raw->basic_cpuid[0][0] >= 2) {
		decode_intel_oldstyle_cache_info(raw, data);
	}
	decode_intel_number_of_cores(raw, data);

	brand = get_brand_code_and_bits(data);
	
	internal->code.intel = brand.code;
	internal->bits = brand.bits;
	
	if (data->flags[CPU_FEATURE_SGX]) {
		// if SGX is indicated by the CPU, verify its presence:
		decode_intel_sgx_features(raw, data);
	}

	return 0;
}
