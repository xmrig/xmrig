#ifndef ARGON2_SSE2_H
#define ARGON2_SSE2_H

#include "core.h"

void xmrig_ar2_fill_segment_sse2(const argon2_instance_t *instance, argon2_position_t position);
int xmrig_ar2_check_sse2(void);

#endif // ARGON2_SSE2_H
