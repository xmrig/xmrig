#ifndef CRYPTONIGHTDARK_H
#define CRYPTONIGHTDARK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void cryptonightdark_hash(const char* input, char* output, uint32_t len, int variant);
void cryptonightdark_fast_hash(const char* input, char* output, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif
