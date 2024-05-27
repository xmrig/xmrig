#ifndef CRYPTONIGHTLITE_H
#define CRYPTONIGHTLITE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void cryptonightlite_hash(const char* input, char* output, uint32_t len, int variant);
void cryptonightlite_fast_hash(const char* input, char* output, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif
