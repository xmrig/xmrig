#ifndef ARGON2_SSSE3_H
#define ARGON2_SSSE3_H

#include "core.h"

void xmrig_ar2_fill_segment_ssse3(const argon2_instance_t *instance, argon2_position_t position);
int xmrig_ar2_check_ssse3(void);

#endif // ARGON2_SSSE3_H
