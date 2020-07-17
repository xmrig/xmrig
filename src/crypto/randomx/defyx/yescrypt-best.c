#ifdef __ARM__
#include "yescrypt-neon.c"
#elif defined __SSE2__
#include "yescrypt-simd.c"
#else
#include "yescrypt-opt.c"
#endif
