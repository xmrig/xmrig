#ifndef ARGON2_IMPL_SELECT_H
#define ARGON2_IMPL_SELECT_H

#include "core.h"

typedef struct Argon2_impl {
    const char *name;
    int (*check)(void);
    void (*fill_segment)(const argon2_instance_t *instance,
                         argon2_position_t position);
} argon2_impl;

typedef struct Argon2_impl_list {
    const argon2_impl *entries;
    size_t count;
} argon2_impl_list;

void argon2_get_impl_list(argon2_impl_list *list);
void fill_segment_default(const argon2_instance_t *instance,
                          argon2_position_t position);

#endif // ARGON2_IMPL_SELECT_H

