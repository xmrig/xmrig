#ifndef CRYPTONIGHTTURTLELITE_H
#define CRYPTONIGHTTURTLELITE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void cryptonightturtlelite_hash(const char* input, char* output, uint32_t len, int variant);
void cryptonightturtlelite_fast_hash(const char* input, char* output, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif
