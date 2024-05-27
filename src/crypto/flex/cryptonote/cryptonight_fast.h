#ifndef CRYPTONIGHTFAST_H
#define CRYPTONIGHTFAST_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void cryptonightfast_hash(const char* input, char* output, uint32_t len, int variant);
void cryptonightfast_fast_hash(const char* input, char* output, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif
