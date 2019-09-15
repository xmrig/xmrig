#include <time.h>
#include <string.h>

#include "impl-select.h"

#include "argon2.h"

#define BENCH_SAMPLES 1024
#define BENCH_MEM_BLOCKS 512

static argon2_impl selected_argon_impl = {
    "default", NULL, fill_segment_default
};

/* the benchmark routine is not thread-safe, so we can use a global var here: */
static block memory[BENCH_MEM_BLOCKS];

static uint64_t benchmark_impl(const argon2_impl *impl) {
    clock_t time;
    unsigned int i;
    uint64_t bench;
    argon2_instance_t instance;
    argon2_position_t pos;

    memset(memory, 0, sizeof(memory));

    instance.version = ARGON2_VERSION_NUMBER;
    instance.memory = memory;
    instance.passes = 1;
    instance.memory_blocks = BENCH_MEM_BLOCKS;
    instance.segment_length = BENCH_MEM_BLOCKS / ARGON2_SYNC_POINTS;
    instance.lane_length = instance.segment_length * ARGON2_SYNC_POINTS;
    instance.lanes = 1;
    instance.threads = 1;
    instance.type = Argon2_i;

    pos.lane = 0;
    pos.pass = 0;
    pos.slice = 0;
    pos.index = 0;

    /* warm-up cache: */
    impl->fill_segment(&instance, pos);

    /* OK, now measure: */
    bench = 0;
    time = clock();
    for (i = 0; i < BENCH_SAMPLES; i++) {
        impl->fill_segment(&instance, pos);
    }
    time = clock() - time;
    bench = (uint64_t)time;
    return bench;
}

void argon2_select_impl()
{
    argon2_impl_list impls;
    unsigned int i;
    const argon2_impl *best_impl = NULL;
    uint64_t best_bench = UINT_MAX;

    argon2_get_impl_list(&impls);

    for (i = 0; i < impls.count; i++) {
        const argon2_impl *impl = &impls.entries[i];
        uint64_t bench;

        if (impl->check != NULL && !impl->check()) {
            continue;
        }

        bench = benchmark_impl(impl);

        if (bench < best_bench) {
            best_bench = bench;
            best_impl = impl;
        }
    }

    if (best_impl != NULL) {
        selected_argon_impl = *best_impl;
    }
}

void fill_segment(const argon2_instance_t *instance, argon2_position_t position)
{
    selected_argon_impl.fill_segment(instance, position);
}

const char *argon2_get_impl_name()
{
    return selected_argon_impl.name;
}


int argon2_select_impl_by_name(const char *name)
{
    argon2_impl_list impls;
    unsigned int i;

    argon2_get_impl_list(&impls);

    for (i = 0; i < impls.count; i++) {
        const argon2_impl *impl = &impls.entries[i];

        if (strcmp(impl->name, name) == 0) {
            selected_argon_impl = *impl;

            return 1;
        }
    }

    return 0;
}
