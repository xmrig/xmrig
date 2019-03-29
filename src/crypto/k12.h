extern "C"
{
#include "k12/KangarooTwelve.h"
}

inline void k12(const uint8_t *input, size_t size, uint8_t *output, cryptonight_ctx **ctx, uint64_t height)
{
  KangarooTwelve((const unsigned char *)input, size, (unsigned char *)output, 32, 0, 0);
}
