#ifndef CRYPTONIGHTTURTLE_H
#define CRYPTONIGHTTURTLE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void cryptonightturtle_hash(const char* input, char* output, uint32_t len, int variant);
void cryptonightturtle_fast_hash(const char* input, char* output, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif
