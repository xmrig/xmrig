#ifndef ARGON2_AVX512F_H
#define ARGON2_AVX512F_H

#include "core.h"

void fill_segment_avx512f(const argon2_instance_t *instance,
                          argon2_position_t position);

int check_avx512f(void);

#endif // ARGON2_AVX512F_H
