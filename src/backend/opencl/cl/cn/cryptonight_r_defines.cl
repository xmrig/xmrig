#ifdef __NV_CL_C_VERSION
#	define SCRATCHPAD_CHUNK(N) (*(__local uint4*)((__local uchar*)(scratchpad_line) + (idx1 ^ (N << 4))))
#else
#	if (STRIDED_INDEX == 0)
#		define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + (idx ^ (N << 4))))
#	elif (STRIDED_INDEX == 1)
#		define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + mul24(as_uint(idx ^ (N << 4)), Threads)))
#	elif (STRIDED_INDEX == 2)
#		define SCRATCHPAD_CHUNK(N) (*(__global uint4*)((__global uchar*)(Scratchpad) + (((idx ^ (N << 4)) % (MEM_CHUNK << 4)) + ((idx ^ (N << 4)) / (MEM_CHUNK << 4)) * WORKSIZE * (MEM_CHUNK << 4))))
#	endif
#endif

#define ROT_BITS 32
#define MEM_CHUNK (1 << MEM_CHUNK_EXPONENT)

#include "wolf-aes.cl"
