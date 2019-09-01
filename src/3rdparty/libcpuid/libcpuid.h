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
#ifndef __LIBCPUID_H__
#define __LIBCPUID_H__
/**
 * \file     libcpuid.h
 * \author   Veselin Georgiev
 * \date     Oct 2008
 * \version  0.4.0
 *
 * Version history:
 *
 * * 0.1.0 (2008-10-15): initial adaptation from wxfractgui sources
 * * 0.1.1 (2009-07-06): Added intel_fn11 fields to cpu_raw_data_t to handle
 *                       new processor topology enumeration required on Core i7
 * * 0.1.2 (2009-09-26): Added support for MSR reading through self-extracting
 *                       kernel driver on Win32.
 * * 0.1.3 (2010-04-20): Added support for greater more accurate CPU clock
 *                       measurements with cpu_clock_by_ic()
 * * 0.2.0 (2011-10-11): Support for AMD Bulldozer CPUs, 128-bit SSE unit size
 *                       checking. A backwards-incompatible change, since the
 *                       sizeof cpu_id_t is now different.
 * * 0.2.1 (2012-05-26): Support for Ivy Bridge, and detecting the presence of
 *                       the RdRand instruction.
 * * 0.2.2 (2015-11-04): Support for newer processors up to Haswell and Vishera.
 *                       Fix clock detection in cpu_clock_by_ic() for Bulldozer.
 *                       More entries supported in cpu_msrinfo().
 *                       *BSD and Solaris support (unofficial).
 * * 0.3.0 (2016-07-09): Support for Skylake; MSR ops in FreeBSD; INFO_VOLTAGE
 *                       for AMD CPUs. Level 4 cache support for Crystalwell
 *                       (a backwards-incompatible change since the sizeof
 *                        cpu_raw_data_t is now different).
 * * 0.4.0 (2016-09-30): Better detection of AMD clock multiplier with msrinfo.
 *                       Support for Intel SGX detection
 *                       (a backwards-incompatible change since the sizeof
 *                        cpu_raw_data_t and cpu_id_t is now different).
 */

/** @mainpage A simple libcpuid introduction
 * 
 * LibCPUID provides CPU identification and access to the CPUID and RDTSC
 * instructions on the x86.
 * <p>
 * To execute CPUID, use \ref cpu_exec_cpuid <br>
 * To execute RDTSC, use \ref cpu_rdtsc <br>
 * To fetch the CPUID info needed for CPU identification, use
 *   \ref cpuid_get_raw_data <br>
 * To make sense of that data (decode, extract features), use \ref cpu_identify <br>
 * To detect the CPU speed, use either \ref cpu_clock, \ref cpu_clock_by_os,
 * \ref cpu_tsc_mark + \ref cpu_tsc_unmark + \ref cpu_clock_by_mark,
 * \ref cpu_clock_measure or \ref cpu_clock_by_ic.
 * Read carefully for pros/cons of each method. <br>
 * 
 * To read MSRs, use \ref cpu_msr_driver_open to get a handle, and then
 * \ref cpu_rdmsr for querying abilities. Some MSR decoding is available on recent
 * CPUs, and can be queried through \ref cpu_msrinfo; the various types of queries
 * are described in \ref cpu_msrinfo_request_t.
 * </p>
 */

/** @defgroup libcpuid LibCPUID
 * @brief LibCPUID provides CPU identification
 @{ */

/* Include some integer type specifications: */
#include "libcpuid_types.h"

/* Some limits and other constants */
#include "libcpuid_constants.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CPU vendor, as guessed from the Vendor String.
 */
typedef enum {
	VENDOR_INTEL = 0,  /*!< Intel CPU */
	VENDOR_AMD,        /*!< AMD CPU */
	VENDOR_CYRIX,      /*!< Cyrix CPU */
	VENDOR_NEXGEN,     /*!< NexGen CPU */
	VENDOR_TRANSMETA,  /*!< Transmeta CPU */
	VENDOR_UMC,        /*!< x86 CPU by UMC */
	VENDOR_CENTAUR,    /*!< x86 CPU by IDT */
	VENDOR_RISE,       /*!< x86 CPU by Rise Technology */
	VENDOR_SIS,        /*!< x86 CPU by SiS */
	VENDOR_NSC,        /*!< x86 CPU by National Semiconductor */
	
	NUM_CPU_VENDORS,   /*!< Valid CPU vendor ids: 0..NUM_CPU_VENDORS - 1 */
	VENDOR_UNKNOWN = -1,
} cpu_vendor_t;
#define NUM_CPU_VENDORS NUM_CPU_VENDORS

/**
 * @brief Contains just the raw CPUID data.
 *
 * This contains only the most basic CPU data, required to do identification
 * and feature recognition. Every processor should be identifiable using this
 * data only.
 */
struct cpu_raw_data_t {
	/** contains results of CPUID for eax = 0, 1, ...*/
	uint32_t basic_cpuid[MAX_CPUID_LEVEL][4];

	/** contains results of CPUID for eax = 0x80000000, 0x80000001, ...*/
	uint32_t ext_cpuid[MAX_EXT_CPUID_LEVEL][4];
	
	/** when the CPU is intel and it supports deterministic cache
	    information: this contains the results of CPUID for eax = 4
	    and ecx = 0, 1, ... */
	uint32_t intel_fn4[MAX_INTELFN4_LEVEL][4];
	
	/** when the CPU is intel and it supports leaf 0Bh (Extended Topology
	    enumeration leaf), this stores the result of CPUID with 
	    eax = 11 and ecx = 0, 1, 2... */
	uint32_t intel_fn11[MAX_INTELFN11_LEVEL][4];
	
	/** when the CPU is intel and supports leaf 12h (SGX enumeration leaf),
	 *  this stores the result of CPUID with eax = 0x12 and
	 *  ecx = 0, 1, 2... */
	uint32_t intel_fn12h[MAX_INTELFN12H_LEVEL][4];

	/** when the CPU is intel and supports leaf 14h (Intel Processor Trace
	 *  capabilities leaf).
	 *  this stores the result of CPUID with eax = 0x12 and
	 *  ecx = 0, 1, 2... */
	uint32_t intel_fn14h[MAX_INTELFN14H_LEVEL][4];
};

/**
 * @brief This contains information about SGX features of the processor
 * Example usage:
 * @code
 * ...
 * struct cpu_raw_data_t raw;
 * struct cpu_id_t id;
 * 
 * if (cpuid_get_raw_data(&raw) == 0 && cpu_identify(&raw, &id) == 0 && id.sgx.present) {
 *   printf("SGX is present.\n");
 *   printf("SGX1 instructions: %s.\n", id.sgx.flags[INTEL_SGX1] ? "present" : "absent");
 *   printf("SGX2 instructions: %s.\n", id.sgx.flags[INTEL_SGX2] ? "present" : "absent");
 *   printf("Max 32-bit enclave size: 2^%d bytes.\n", id.sgx.max_enclave_32bit);
 *   printf("Max 64-bit enclave size: 2^%d bytes.\n", id.sgx.max_enclave_64bit);
 *   for (int i = 0; i < id.sgx.num_epc_sections; i++) {
 *     struct cpu_epc_t epc = cpuid_get_epc(i, NULL);
 *     printf("EPC section #%d: address = %x, size = %d bytes.\n", epc.address, epc.size);
 *   }
 * } else {
 *   printf("SGX is not present.\n");
 * }
 * @endcode
 */ 
struct cpu_sgx_t {
	/** Whether SGX is present (boolean) */
	uint32_t present;
	
	/** Max enclave size in 32-bit mode. This is a power-of-two value:
	 *  if it is "31", then the max enclave size is 2^31 bytes (2 GiB).
	 */
	uint8_t max_enclave_32bit;
	
	/** Max enclave size in 64-bit mode. This is a power-of-two value:
	 *  if it is "36", then the max enclave size is 2^36 bytes (64 GiB).
	 */
	uint8_t max_enclave_64bit;
	
	/**
	 * contains SGX feature flags. See the \ref cpu_sgx_feature_t
	 * "INTEL_SGX*" macros below.
	 */
	uint8_t flags[SGX_FLAGS_MAX];
	
	/** number of Enclave Page Cache (EPC) sections. Info for each
	 *  section is available through the \ref cpuid_get_epc() function
	 */
	int num_epc_sections;
	
	/** bit vector of the supported extended  features that can be written
	 *  to the MISC region of the SSA (Save State Area)
	 */ 
	uint32_t misc_select;
	
	/** a bit vector of the attributes that can be set to SECS.ATTRIBUTES
	 *  via ECREATE. Corresponds to bits 0-63 (incl.) of SECS.ATTRIBUTES.
	 */ 
	uint64_t secs_attributes;
	
	/** a bit vector of the bits that can be set in the XSAVE feature
	 *  request mask; Corresponds to bits 64-127 of SECS.ATTRIBUTES.
	 */
	uint64_t secs_xfrm;
};

/**
 * @brief This contains the recognized CPU features/info
 */
struct cpu_id_t {
	/** contains the CPU vendor string, e.g. "GenuineIntel" */
	char vendor_str[VENDOR_STR_MAX];
	
	/** contains the brand string, e.g. "Intel(R) Xeon(TM) CPU 2.40GHz" */
	char brand_str[BRAND_STR_MAX];
	
	/** contains the recognized CPU vendor */
	cpu_vendor_t vendor;
	
	/**
	 * contain CPU flags. Used to test for features. See
	 * the \ref cpu_feature_t "CPU_FEATURE_*" macros below.
	 * @see Features
	 */
	uint8_t flags[CPU_FLAGS_MAX];
	
	/** CPU family */
	int32_t family;
	
	/** CPU model */
	int32_t model;
	
	/** CPU stepping */
	int32_t stepping;
	
	/** CPU extended family */
	int32_t ext_family;
	
	/** CPU extended model */
	int32_t ext_model;
	
	/** Number of CPU cores on the current processor */
	int32_t num_cores;
	
	/**
	 * Number of logical processors on the current processor.
	 * Could be more than the number of physical cores,
	 * e.g. when the processor has HyperThreading.
	 */
	int32_t num_logical_cpus;
	
	/**
	 * The total number of logical processors.
	 * The same value is availabe through \ref cpuid_get_total_cpus.
	 *
	 * This is num_logical_cpus * {total physical processors in the system}
	 * (but only on a real system, under a VM this number may be lower).
	 *
	 * If you're writing a multithreaded program and you want to run it on
	 * all CPUs, this is the number of threads you need.
	 *
	 * @note in a VM, this will exactly match the number of CPUs set in
	 *       the VM's configuration.
	 *
	 */
	int32_t total_logical_cpus;
	
	/**
	 * L1 data cache size in KB. Could be zero, if the CPU lacks cache.
	 * If the size cannot be determined, it will be -1.
	 */
	int32_t l1_data_cache;
	
	/**
	 * L1 instruction cache size in KB. Could be zero, if the CPU lacks
	 * cache. If the size cannot be determined, it will be -1.
	 * @note On some Intel CPUs, whose instruction cache is in fact
	 * a trace cache, the size will be expressed in K uOps.
	 */
	int32_t l1_instruction_cache;
	
	/**
	 * L2 cache size in KB. Could be zero, if the CPU lacks L2 cache.
	 * If the size of the cache could not be determined, it will be -1
	 */
	int32_t l2_cache;
	
	/** L3 cache size in KB. Zero on most systems */
	int32_t l3_cache;

	/** L4 cache size in KB. Zero on most systems */
	int32_t l4_cache;
	
	/** Cache associativity for the L1 data cache. -1 if undetermined */
	int32_t l1_assoc;
	
	/** Cache associativity for the L2 cache. -1 if undetermined */
	int32_t l2_assoc;
	
	/** Cache associativity for the L3 cache. -1 if undetermined */
	int32_t l3_assoc;

	/** Cache associativity for the L4 cache. -1 if undetermined */
	int32_t l4_assoc;
	
	/** Cache-line size for L1 data cache. -1 if undetermined */
	int32_t l1_cacheline;
	
	/** Cache-line size for L2 cache. -1 if undetermined */
	int32_t l2_cacheline;
	
	/** Cache-line size for L3 cache. -1 if undetermined */
	int32_t l3_cacheline;
	
	/** Cache-line size for L4 cache. -1 if undetermined */
	int32_t l4_cacheline;

	/**
	 * The brief and human-friendly CPU codename, which was recognized.<br>
	 * Examples:
	 * @code
	 * +--------+--------+-------+-------+-------+---------------------------------------+-----------------------+
	 * | Vendor | Family | Model | Step. | Cache |       Brand String                    | cpu_id_t.cpu_codename |
	 * +--------+--------+-------+-------+-------+---------------------------------------+-----------------------+
	 * | AMD    |      6 |     8 |     0 |   256 | (not available - will be ignored)     | "K6-2"                |
	 * | Intel  |     15 |     2 |     5 |   512 | "Intel(R) Xeon(TM) CPU 2.40GHz"       | "Xeon (Prestonia)"    |
	 * | Intel  |      6 |    15 |    11 |  4096 | "Intel(R) Core(TM)2 Duo CPU E6550..." | "Conroe (Core 2 Duo)" |
	 * | AMD    |     15 |    35 |     2 |  1024 | "Dual Core AMD Opteron(tm) Proces..." | "Opteron (Dual Core)" |
	 * +--------+--------+-------+-------+-------+---------------------------------------+-----------------------+
	 * @endcode
	 */
	char cpu_codename[64];
	
	/** SSE execution unit size (64 or 128; -1 if N/A) */
	int32_t sse_size;
	
	/**
	 * contain miscellaneous detection information. Used to test about specifics of
	 * certain detected features. See \ref cpu_hint_t "CPU_HINT_*" macros below.
	 * @see Hints
	 */
	uint8_t detection_hints[CPU_HINTS_MAX];
	
	/** contains information about SGX features if the processor, if present */
	struct cpu_sgx_t sgx;
};

/**
 * @brief CPU feature identifiers
 *
 * Usage:
 * @code
 * ...
 * struct cpu_raw_data_t raw;
 * struct cpu_id_t id;
 * if (cpuid_get_raw_data(&raw) == 0 && cpu_identify(&raw, &id) == 0) {
 *     if (id.flags[CPU_FEATURE_SSE2]) {
 *         // The CPU has SSE2...
 *         ...
 *     } else {
 *         // no SSE2
 *     }
 * } else {
 *   // processor cannot be determined.
 * }
 * @endcode
 */
typedef enum {
	CPU_FEATURE_FPU = 0,	/*!< Floating point unit */
	CPU_FEATURE_VME,	/*!< Virtual mode extension */
	CPU_FEATURE_DE,		/*!< Debugging extension */
	CPU_FEATURE_PSE,	/*!< Page size extension */
	CPU_FEATURE_TSC,	/*!< Time-stamp counter */
	CPU_FEATURE_MSR,	/*!< Model-specific regsisters, RDMSR/WRMSR supported */
	CPU_FEATURE_PAE,	/*!< Physical address extension */
	CPU_FEATURE_MCE,	/*!< Machine check exception */
	CPU_FEATURE_CX8,	/*!< CMPXCHG8B instruction supported */
	CPU_FEATURE_APIC,	/*!< APIC support */
	CPU_FEATURE_MTRR,	/*!< Memory type range registers */
	CPU_FEATURE_SEP,	/*!< SYSENTER / SYSEXIT instructions supported */
	CPU_FEATURE_PGE,	/*!< Page global enable */
	CPU_FEATURE_MCA,	/*!< Machine check architecture */
	CPU_FEATURE_CMOV,	/*!< CMOVxx instructions supported */
	CPU_FEATURE_PAT,	/*!< Page attribute table */
	CPU_FEATURE_PSE36,	/*!< 36-bit page address extension */
	CPU_FEATURE_PN,		/*!< Processor serial # implemented (Intel P3 only) */
	CPU_FEATURE_CLFLUSH,	/*!< CLFLUSH instruction supported */
	CPU_FEATURE_DTS,	/*!< Debug store supported */
	CPU_FEATURE_ACPI,	/*!< ACPI support (power states) */
	CPU_FEATURE_MMX,	/*!< MMX instruction set supported */
	CPU_FEATURE_FXSR,	/*!< FXSAVE / FXRSTOR supported */
	CPU_FEATURE_SSE,	/*!< Streaming-SIMD Extensions (SSE) supported */
	CPU_FEATURE_SSE2,	/*!< SSE2 instructions supported */
	CPU_FEATURE_SS,		/*!< Self-snoop */
	CPU_FEATURE_HT,		/*!< Hyper-threading supported (but might be disabled) */
	CPU_FEATURE_TM,		/*!< Thermal monitor */
	CPU_FEATURE_IA64,	/*!< IA64 supported (Itanium only) */
	CPU_FEATURE_PBE,	/*!< Pending-break enable */
	CPU_FEATURE_PNI,	/*!< PNI (SSE3) instructions supported */
	CPU_FEATURE_PCLMUL,	/*!< PCLMULQDQ instruction supported */
	CPU_FEATURE_DTS64,	/*!< 64-bit Debug store supported */
	CPU_FEATURE_MONITOR,	/*!< MONITOR / MWAIT supported */
	CPU_FEATURE_DS_CPL,	/*!< CPL Qualified Debug Store */
	CPU_FEATURE_VMX,	/*!< Virtualization technology supported */
	CPU_FEATURE_SMX,	/*!< Safer mode exceptions */
	CPU_FEATURE_EST,	/*!< Enhanced SpeedStep */
	CPU_FEATURE_TM2,	/*!< Thermal monitor 2 */
	CPU_FEATURE_SSSE3,	/*!< SSSE3 instructionss supported (this is different from SSE3!) */
	CPU_FEATURE_CID,	/*!< Context ID supported */
	CPU_FEATURE_CX16,	/*!< CMPXCHG16B instruction supported */
	CPU_FEATURE_XTPR,	/*!< Send Task Priority Messages disable */
	CPU_FEATURE_PDCM,	/*!< Performance capabilities MSR supported */
	CPU_FEATURE_DCA,	/*!< Direct cache access supported */
	CPU_FEATURE_SSE4_1,	/*!< SSE 4.1 instructions supported */
	CPU_FEATURE_SSE4_2,	/*!< SSE 4.2 instructions supported */
	CPU_FEATURE_SYSCALL,	/*!< SYSCALL / SYSRET instructions supported */
	CPU_FEATURE_XD,		/*!< Execute disable bit supported */
	CPU_FEATURE_MOVBE,	/*!< MOVBE instruction supported */
	CPU_FEATURE_POPCNT,	/*!< POPCNT instruction supported */
	CPU_FEATURE_AES,	/*!< AES* instructions supported */
	CPU_FEATURE_XSAVE,	/*!< XSAVE/XRSTOR/etc instructions supported */
	CPU_FEATURE_OSXSAVE,	/*!< non-privileged copy of OSXSAVE supported */
	CPU_FEATURE_AVX,	/*!< Advanced vector extensions supported */
	CPU_FEATURE_MMXEXT,	/*!< AMD MMX-extended instructions supported */
	CPU_FEATURE_3DNOW,	/*!< AMD 3DNow! instructions supported */
	CPU_FEATURE_3DNOWEXT,	/*!< AMD 3DNow! extended instructions supported */
	CPU_FEATURE_NX,		/*!< No-execute bit supported */
	CPU_FEATURE_FXSR_OPT,	/*!< FFXSR: FXSAVE and FXRSTOR optimizations */
	CPU_FEATURE_RDTSCP,	/*!< RDTSCP instruction supported (AMD-only) */
	CPU_FEATURE_LM,		/*!< Long mode (x86_64/EM64T) supported */
	CPU_FEATURE_LAHF_LM,	/*!< LAHF/SAHF supported in 64-bit mode */
	CPU_FEATURE_CMP_LEGACY,	/*!< core multi-processing legacy mode */
	CPU_FEATURE_SVM,	/*!< AMD Secure virtual machine */
	CPU_FEATURE_ABM,	/*!< LZCNT instruction support */
	CPU_FEATURE_MISALIGNSSE,/*!< Misaligned SSE supported */
	CPU_FEATURE_SSE4A,	/*!< SSE 4a from AMD */
	CPU_FEATURE_3DNOWPREFETCH,	/*!< PREFETCH/PREFETCHW support */
	CPU_FEATURE_OSVW,	/*!< OS Visible Workaround (AMD) */
	CPU_FEATURE_IBS,	/*!< Instruction-based sampling */
	CPU_FEATURE_SSE5,	/*!< SSE 5 instructions supported (deprecated, will never be 1) */
	CPU_FEATURE_SKINIT,	/*!< SKINIT / STGI supported */
	CPU_FEATURE_WDT,	/*!< Watchdog timer support */
	CPU_FEATURE_TS,		/*!< Temperature sensor */
	CPU_FEATURE_FID,	/*!< Frequency ID control */
	CPU_FEATURE_VID,	/*!< Voltage ID control */
	CPU_FEATURE_TTP,	/*!< THERMTRIP */
	CPU_FEATURE_TM_AMD,	/*!< AMD-specified hardware thermal control */
	CPU_FEATURE_STC,	/*!< Software thermal control */
	CPU_FEATURE_100MHZSTEPS,/*!< 100 MHz multiplier control */
	CPU_FEATURE_HWPSTATE,	/*!< Hardware P-state control */
	CPU_FEATURE_CONSTANT_TSC,	/*!< TSC ticks at constant rate */
	CPU_FEATURE_XOP,	/*!< The XOP instruction set (same as the old CPU_FEATURE_SSE5) */
	CPU_FEATURE_FMA3,	/*!< The FMA3 instruction set */
	CPU_FEATURE_FMA4,	/*!< The FMA4 instruction set */
	CPU_FEATURE_TBM,	/*!< Trailing bit manipulation instruction support */
	CPU_FEATURE_F16C,	/*!< 16-bit FP convert instruction support */
	CPU_FEATURE_RDRAND,     /*!< RdRand instruction */
	CPU_FEATURE_X2APIC,     /*!< x2APIC, APIC_BASE.EXTD, MSRs 0000_0800h...0000_0BFFh 64-bit ICR (+030h but not +031h), no DFR (+00Eh), SELF_IPI (+040h) also see standard level 0000_000Bh */
	CPU_FEATURE_CPB,	/*!< Core performance boost */
	CPU_FEATURE_APERFMPERF,	/*!< MPERF/APERF MSRs support */
	CPU_FEATURE_PFI,	/*!< Processor Feedback Interface support */
	CPU_FEATURE_PA,		/*!< Processor accumulator */
	CPU_FEATURE_AVX2,	/*!< AVX2 instructions */
	CPU_FEATURE_BMI1,	/*!< BMI1 instructions */
	CPU_FEATURE_BMI2,	/*!< BMI2 instructions */
	CPU_FEATURE_HLE,	/*!< Hardware Lock Elision prefixes */
	CPU_FEATURE_RTM,	/*!< Restricted Transactional Memory instructions */
	CPU_FEATURE_AVX512F,	/*!< AVX-512 Foundation */
	CPU_FEATURE_AVX512DQ,	/*!< AVX-512 Double/Quad granular insns */
	CPU_FEATURE_AVX512PF,	/*!< AVX-512 Prefetch */
	CPU_FEATURE_AVX512ER,	/*!< AVX-512 Exponential/Reciprocal */
	CPU_FEATURE_AVX512CD,	/*!< AVX-512 Conflict detection */
	CPU_FEATURE_SHA_NI,	/*!< SHA-1/SHA-256 instructions */
	CPU_FEATURE_AVX512BW,	/*!< AVX-512 Byte/Word granular insns */
	CPU_FEATURE_AVX512VL,	/*!< AVX-512 128/256 vector length extensions */
	CPU_FEATURE_SGX,	/*!< SGX extensions. Non-autoritative, check cpu_id_t::sgx::present to verify presence */
	CPU_FEATURE_RDSEED,	/*!< RDSEED instruction */
	CPU_FEATURE_ADX,	/*!< ADX extensions (arbitrary precision) */
	/* termination: */
	NUM_CPU_FEATURES,
} cpu_feature_t;

/**
 * @brief CPU detection hints identifiers
 *
 * Usage: similar to the flags usage
 */
typedef enum {
	CPU_HINT_SSE_SIZE_AUTH = 0,	/*!< SSE unit size is authoritative (not only a Family/Model guesswork, but based on an actual CPUID bit) */
	/* termination */
	NUM_CPU_HINTS,
} cpu_hint_t;

/**
 * @brief SGX features flags
 * \see cpu_sgx_t
 *
 * Usage:
 * @code
 * ...
 * struct cpu_raw_data_t raw;
 * struct cpu_id_t id;
 * if (cpuid_get_raw_data(&raw) == 0 && cpu_identify(&raw, &id) == 0 && id.sgx.present) {
 *     if (id.sgx.flags[INTEL_SGX1])
 *         // The CPU has SGX1 instructions support...
 *         ...
 *     } else {
 *         // no SGX
 *     }
 * } else {
 *   // processor cannot be determined.
 * }
 * @endcode
 */
 
typedef enum {
	INTEL_SGX1,		/*!< SGX1 instructions support */
	INTEL_SGX2,		/*!< SGX2 instructions support */
	
	/* termination: */
	NUM_SGX_FEATURES,
} cpu_sgx_feature_t;

/**
 * @brief Describes common library error codes
 */
typedef enum {
	ERR_OK       =  0,	/*!< No error */
	ERR_NO_CPUID = -1,	/*!< CPUID instruction is not supported */
	ERR_NO_RDTSC = -2,	/*!< RDTSC instruction is not supported */
	ERR_NO_MEM   = -3,	/*!< Memory allocation failed */
	ERR_OPEN     = -4,	/*!< File open operation failed */
	ERR_BADFMT   = -5,	/*!< Bad file format */
	ERR_NOT_IMP  = -6,	/*!< Not implemented */
	ERR_CPU_UNKN = -7,	/*!< Unsupported processor */
	ERR_NO_RDMSR = -8,	/*!< RDMSR instruction is not supported */
	ERR_NO_DRIVER= -9,	/*!< RDMSR driver error (generic) */
	ERR_NO_PERMS = -10,	/*!< No permissions to install RDMSR driver */
	ERR_EXTRACT  = -11,	/*!< Cannot extract RDMSR driver (read only media?) */
	ERR_HANDLE   = -12,	/*!< Bad handle */
	ERR_INVMSR   = -13,	/*!< Invalid MSR */
	ERR_INVCNB   = -14,	/*!< Invalid core number */
	ERR_HANDLE_R = -15,	/*!< Error on handle read */
	ERR_INVRANGE = -16,	/*!< Invalid given range */
} cpu_error_t;

/**
 * @brief Internal structure, used in cpu_tsc_mark, cpu_tsc_unmark and
 *        cpu_clock_by_mark
 */
struct cpu_mark_t {
	uint64_t tsc;		/*!< Time-stamp from RDTSC */
	uint64_t sys_clock;	/*!< In microsecond resolution */
};

/**
 * @brief Returns the total number of logical CPU threads (even if CPUID is not present).
 *
 * Under VM, this number (and total_logical_cpus, since they are fetched with the same code)
 * may be nonsensical, i.e. might not equal NumPhysicalCPUs*NumCoresPerCPU*HyperThreading.
 * This is because no matter how many logical threads the host machine has, you may limit them
 * in the VM to any number you like. **This** is the number returned by cpuid_get_total_cpus().
 *
 * @returns Number of logical CPU threads available. Equals the \ref cpu_id_t::total_logical_cpus.
 */
int cpuid_get_total_cpus(void);

/**
 * @brief Checks if the CPUID instruction is supported
 * @retval 1 if CPUID is present
 * @retval 0 the CPU doesn't have CPUID.
 */
int cpuid_present(void);

/**
 * @brief Executes the CPUID instruction
 * @param eax - the value of the EAX register when executing CPUID
 * @param regs - the results will be stored here. regs[0] = EAX, regs[1] = EBX, ...
 * @note CPUID will be executed with EAX set to the given value and EBX, ECX,
 *       EDX set to zero.
 */
void cpu_exec_cpuid(uint32_t eax, uint32_t* regs);

/**
 * @brief Executes the CPUID instruction with the given input registers
 * @note This is just a bit more generic version of cpu_exec_cpuid - it allows
 *       you to control all the registers.
 * @param regs - Input/output. Prior to executing CPUID, EAX, EBX, ECX and
 *               EDX will be set to regs[0], regs[1], regs[2] and regs[3].
 *               After CPUID, this array will contain the results.
 */
void cpu_exec_cpuid_ext(uint32_t* regs);

/**
 * @brief Obtains the raw CPUID data from the current CPU
 * @param data - a pointer to cpu_raw_data_t structure
 * @returns zero if successful, and some negative number on error.
 *          The error message can be obtained by calling \ref cpuid_error.
 *          @see cpu_error_t
 */
int cpuid_get_raw_data(struct cpu_raw_data_t* data);

/**
 * @brief Identifies the CPU
 * @param raw - Input - a pointer to the raw CPUID data, which is obtained
 *              either by cpuid_get_raw_data or cpuid_deserialize_raw_data.
 *              Can also be NULL, in which case the functions calls
 *              cpuid_get_raw_data itself.
 * @param data - Output - the decoded CPU features/info is written here.
 * @note The function will not fail, even if some of the information
 *       cannot be obtained. Even when the CPU is new and thus unknown to
 *       libcpuid, some generic info, such as "AMD K9 family CPU" will be
 *       written to data.cpu_codename, and most other things, such as the
 *       CPU flags, cache sizes, etc. should be detected correctly anyway.
 *       However, the function CAN fail, if the CPU is completely alien to
 *       libcpuid.
 * @note While cpu_identify() and cpuid_get_raw_data() are fast for most
 *       purposes, running them several thousand times per second can hamper
 *       performance significantly. Specifically, avoid writing "cpu feature
 *       checker" wrapping function, which calls cpu_identify and returns the
 *       value of some flag, if that function is going to be called frequently.
 * @returns zero if successful, and some negative number on error.
 *          The error message can be obtained by calling \ref cpuid_error.
 *          @see cpu_error_t
 */
int cpu_identify(struct cpu_raw_data_t* raw, struct cpu_id_t* data);

/**
 * @brief The return value of cpuid_get_epc().
 * @details
 * Describes an EPC (Enclave Page Cache) layout (physical address and size).
 * A CPU may have one or more EPC areas, and information about each is
 * fetched via \ref cpuid_get_epc.
 */ 
struct cpu_epc_t {
	uint64_t start_addr;
	uint64_t length;
};

/**
 * @brief Fetches information about an EPC (Enclave Page Cache) area.
 * @param index - zero-based index, valid range [0..cpu_id_t.egx.num_epc_sections)
 * @param raw   - a pointer to fetched raw CPUID data. Needed only for testing,
 *                you can safely pass NULL here (if you pass a real structure,
 *                it will be used for fetching the leaf 12h data if index < 2;
 *                otherwise the real CPUID instruction will be used).
 * @returns the requested data. If the CPU doesn't support SGX, or if
 *          index >= cpu_id_t.egx.num_epc_sections, both fields of the returned
 *          structure will be zeros.
 */
struct cpu_epc_t cpuid_get_epc(int index, const struct cpu_raw_data_t* raw);

/**
 * @brief Returns the libcpuid version
 *
 * @returns the string representation of the libcpuid version, like "0.1.1"
 */
const char* cpuid_lib_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif


/** @} */

#endif /* __LIBCPUID_H__ */
