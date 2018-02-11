#ifndef __LIBCPUID_H__
#define __LIBCPUID_H__
#include "libcpuid_types.h"

#include "libcpuid_constants.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	VENDOR_INTEL = 0,  
	VENDOR_AMD,        
	VENDOR_CYRIX,      
	VENDOR_NEXGEN,     
	VENDOR_TRANSMETA,  
	VENDOR_UMC,        
	VENDOR_CENTAUR,    
	VENDOR_RISE,       
	VENDOR_SIS,        
	VENDOR_NSC,        
	
	NUM_CPU_VENDORS,   
	VENDOR_UNKNOWN = -1,
} cpu_vendor_t;
#define NUM_CPU_VENDORS NUM_CPU_VENDORS
struct cpu_raw_data_t {
	
	uint32_t basic_cpuid[MAX_CPUID_LEVEL][4];

	
	uint32_t ext_cpuid[MAX_EXT_CPUID_LEVEL][4];
	
	uint32_t intel_fn4[MAX_INTELFN4_LEVEL][4];
	uint32_t intel_fn11[MAX_INTELFN11_LEVEL][4];
	uint32_t intel_fn12h[MAX_INTELFN12H_LEVEL][4];
	uint32_t intel_fn14h[MAX_INTELFN14H_LEVEL][4];
};
struct cpu_sgx_t {
	
	uint32_t present;
	uint8_t max_enclave_32bit;
	uint8_t max_enclave_64bit;
	uint8_t flags[SGX_FLAGS_MAX];
	int num_epc_sections;
	uint32_t misc_select;
	uint64_t secs_attributes;
	uint64_t secs_xfrm;
};
struct cpu_id_t {
	
	char vendor_str[VENDOR_STR_MAX];
	
	
	char brand_str[BRAND_STR_MAX];
	
	
	cpu_vendor_t vendor;
	uint8_t flags[CPU_FLAGS_MAX];
	
	
	int32_t family;
	
	
	int32_t model;
	
	
	int32_t stepping;
	
	
	int32_t ext_family;
	
	
	int32_t ext_model;
	
	
	int32_t num_cores;
	int32_t num_logical_cpus;
	int32_t total_logical_cpus;
	int32_t l1_data_cache;
	int32_t l1_instruction_cache;
	int32_t l2_cache;
	
	
	int32_t l3_cache;

	
	int32_t l4_cache;
	
	
	int32_t l1_assoc;
	
	
	int32_t l2_assoc;
	
	
	int32_t l3_assoc;

	
	int32_t l4_assoc;
	
	
	int32_t l1_cacheline;
	
	
	int32_t l2_cacheline;
	
	
	int32_t l3_cacheline;
	
	
	int32_t l4_cacheline;
	char cpu_codename[64];
	
	
	int32_t sse_size;
	uint8_t detection_hints[CPU_HINTS_MAX];
	
	
	struct cpu_sgx_t sgx;
};
typedef enum {
	CPU_FEATURE_FPU = 0,	
	CPU_FEATURE_VME,	
	CPU_FEATURE_DE,		
	CPU_FEATURE_PSE,	
	CPU_FEATURE_TSC,	
	CPU_FEATURE_MSR,	
	CPU_FEATURE_PAE,	
	CPU_FEATURE_MCE,	
	CPU_FEATURE_CX8,	
	CPU_FEATURE_APIC,	
	CPU_FEATURE_MTRR,	
	CPU_FEATURE_SEP,	
	CPU_FEATURE_PGE,	
	CPU_FEATURE_MCA,	
	CPU_FEATURE_CMOV,	
	CPU_FEATURE_PAT,	
	CPU_FEATURE_PSE36,	
	CPU_FEATURE_PN,		
	CPU_FEATURE_CLFLUSH,	
	CPU_FEATURE_DTS,	
	CPU_FEATURE_ACPI,	
	CPU_FEATURE_MMX,	
	CPU_FEATURE_FXSR,	
	CPU_FEATURE_SSE,	
	CPU_FEATURE_SSE2,	
	CPU_FEATURE_SS,		
	CPU_FEATURE_HT,		
	CPU_FEATURE_TM,		
	CPU_FEATURE_IA64,	
	CPU_FEATURE_PBE,	
	CPU_FEATURE_PNI,	
	CPU_FEATURE_PCLMUL,	
	CPU_FEATURE_DTS64,	
	CPU_FEATURE_MONITOR,	
	CPU_FEATURE_DS_CPL,	
	CPU_FEATURE_VMX,	
	CPU_FEATURE_SMX,	
	CPU_FEATURE_EST,	
	CPU_FEATURE_TM2,	
	CPU_FEATURE_SSSE3,	
	CPU_FEATURE_CID,	
	CPU_FEATURE_CX16,	
	CPU_FEATURE_XTPR,	
	CPU_FEATURE_PDCM,	
	CPU_FEATURE_DCA,	
	CPU_FEATURE_SSE4_1,	
	CPU_FEATURE_SSE4_2,	
	CPU_FEATURE_SYSCALL,	
	CPU_FEATURE_XD,		
	CPU_FEATURE_MOVBE,	
	CPU_FEATURE_POPCNT,	
	CPU_FEATURE_AES,	
	CPU_FEATURE_XSAVE,	
	CPU_FEATURE_OSXSAVE,	
	CPU_FEATURE_AVX,	
	CPU_FEATURE_MMXEXT,	
	CPU_FEATURE_3DNOW,	
	CPU_FEATURE_3DNOWEXT,	
	CPU_FEATURE_NX,		
	CPU_FEATURE_FXSR_OPT,	
	CPU_FEATURE_RDTSCP,	
	CPU_FEATURE_LM,		
	CPU_FEATURE_LAHF_LM,	
	CPU_FEATURE_CMP_LEGACY,	
	CPU_FEATURE_SVM,	
	CPU_FEATURE_ABM,	
	CPU_FEATURE_MISALIGNSSE,
	CPU_FEATURE_SSE4A,	
	CPU_FEATURE_3DNOWPREFETCH,	
	CPU_FEATURE_OSVW,	
	CPU_FEATURE_IBS,	
	CPU_FEATURE_SSE5,	
	CPU_FEATURE_SKINIT,	
	CPU_FEATURE_WDT,	
	CPU_FEATURE_TS,		
	CPU_FEATURE_FID,	
	CPU_FEATURE_VID,	
	CPU_FEATURE_TTP,	
	CPU_FEATURE_TM_AMD,	
	CPU_FEATURE_STC,	
	CPU_FEATURE_100MHZSTEPS,
	CPU_FEATURE_HWPSTATE,	
	CPU_FEATURE_CONSTANT_TSC,	
	CPU_FEATURE_XOP,	
	CPU_FEATURE_FMA3,	
	CPU_FEATURE_FMA4,	
	CPU_FEATURE_TBM,	
	CPU_FEATURE_F16C,	
	CPU_FEATURE_RDRAND,     
	CPU_FEATURE_X2APIC,     
	CPU_FEATURE_CPB,	
	CPU_FEATURE_APERFMPERF,	
	CPU_FEATURE_PFI,	
	CPU_FEATURE_PA,		
	CPU_FEATURE_AVX2,	
	CPU_FEATURE_BMI1,	
	CPU_FEATURE_BMI2,	
	CPU_FEATURE_HLE,	
	CPU_FEATURE_RTM,	
	CPU_FEATURE_AVX512F,	
	CPU_FEATURE_AVX512DQ,	
	CPU_FEATURE_AVX512PF,	
	CPU_FEATURE_AVX512ER,	
	CPU_FEATURE_AVX512CD,	
	CPU_FEATURE_SHA_NI,	
	CPU_FEATURE_AVX512BW,	
	CPU_FEATURE_AVX512VL,	
	CPU_FEATURE_SGX,	
	CPU_FEATURE_RDSEED,	
	CPU_FEATURE_ADX,	
	
	NUM_CPU_FEATURES,
} cpu_feature_t;
typedef enum {
	CPU_HINT_SSE_SIZE_AUTH = 0,	
	
	NUM_CPU_HINTS,
} cpu_hint_t;
typedef enum {
	INTEL_SGX1,		
	INTEL_SGX2,		
	
	
	NUM_SGX_FEATURES,
} cpu_sgx_feature_t;

typedef enum {
	ERR_OK       =  0,	
	ERR_NO_CPUID = -1,	
	ERR_NO_RDTSC = -2,	
	ERR_NO_MEM   = -3,	
	ERR_OPEN     = -4,	
	ERR_BADFMT   = -5,	
	ERR_NOT_IMP  = -6,	
	ERR_CPU_UNKN = -7,	
	ERR_NO_RDMSR = -8,	
	ERR_NO_DRIVER= -9,	
	ERR_NO_PERMS = -10,	
	ERR_EXTRACT  = -11,	
	ERR_HANDLE   = -12,	
	ERR_INVMSR   = -13,	
	ERR_INVCNB   = -14,	
	ERR_HANDLE_R = -15,	
	ERR_INVRANGE = -16,	
} cpu_error_t;

struct cpu_mark_t {
	uint64_t tsc;		
	uint64_t sys_clock;	
};

int cpuid_get_total_cpus(void);
int cpuid_present(void);
void cpu_exec_cpuid(uint32_t eax, uint32_t* regs);
void cpu_exec_cpuid_ext(uint32_t* regs);
int cpuid_get_raw_data(struct cpu_raw_data_t* data);
int cpu_identify(struct cpu_raw_data_t* raw, struct cpu_id_t* data);
struct cpu_epc_t {
	uint64_t start_addr;
	uint64_t length;
};
struct cpu_epc_t cpuid_get_epc(int index, const struct cpu_raw_data_t* raw);
const char* cpuid_lib_version(void);

#ifdef __cplusplus
}; 
#endif

#endif 
