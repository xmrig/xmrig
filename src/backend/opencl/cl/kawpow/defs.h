#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif

#ifndef GROUP_SIZE
#define GROUP_SIZE 256
#endif
#define GROUP_SHARE (GROUP_SIZE / 16)

typedef unsigned int       uint32_t;
typedef unsigned long      uint64_t;
#define ROTL32(x, n) rotate((x), (uint32_t)(n))
#define ROTR32(x, n) rotate((x), (uint32_t)(32-n))

#define PROGPOW_LANES           16
#define PROGPOW_REGS            32
#define PROGPOW_DAG_LOADS       4
#define PROGPOW_CACHE_WORDS     4096
#define PROGPOW_CNT_DAG         64
#define PROGPOW_CNT_MATH        18

#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_NVIDIA 1
#define OPENCL_PLATFORM_AMD 2
#define OPENCL_PLATFORM_CLOVER 3

#ifndef MAX_OUTPUTS
#define MAX_OUTPUTS 63U
#endif

#ifndef PLATFORM
#define PLATFORM OPENCL_PLATFORM_AMD
#endif

#define HASHES_PER_GROUP (GROUP_SIZE / PROGPOW_LANES)

#define FNV_PRIME 0x1000193
#define FNV_OFFSET_BASIS 0x811c9dc5
