#ifndef ARGON2_AVX2_H
#define ARGON2_AVX2_H

#include "core.h"

void xmrig_ar2_fill_segment_avx2(const argon2_instance_t *instance, argon2_position_t position);
int xmrig_ar2_check_avx2(void);

#endif // ARGON2_AVX2_H
