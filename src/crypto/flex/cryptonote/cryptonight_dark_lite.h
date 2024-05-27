#ifndef CRYPTONIGHTDARKLITE_H
#define CRYPTONIGHTDARKLITE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void cryptonightdarklite_hash(const char* input, char* output, uint32_t len, int variant);
void cryptonightdarklite_fast_hash(const char* input, char* output, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif
