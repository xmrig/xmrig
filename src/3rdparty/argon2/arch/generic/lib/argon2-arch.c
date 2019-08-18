#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "impl-select.h"

#define rotr64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

#include "argon2-template-64.h"

void fill_segment_default(const argon2_instance_t *instance,
                          argon2_position_t position)
{
    fill_segment_64(instance, position);
}

void argon2_get_impl_list(argon2_impl_list *list)
{
    list->count = 0;
}
