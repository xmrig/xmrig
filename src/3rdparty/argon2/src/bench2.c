#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "argon2.h"

#include "timing.h"

#define ARGON2_BLOCK_SIZE 1024

#define BENCH_MAX_T_COST 16
#define BENCH_MAX_M_COST (1024 * 1024)
#define BENCH_MAX_THREADS 8
#define BENCH_MIN_PASSES (1024 * 1024)
#define BENCH_MAX_SAMPLES 128

#define BENCH_OUTLEN 16
#define BENCH_INLEN 16

static double pick_min(const double *samples, size_t count)
{
    size_t i;
    double min = INFINITY;
    for (i = 0; i < count; i++) {
        if (samples[i] < min) {
            min = samples[i];
        }
    }
    return min;
}

static int benchmark(void *memory, size_t memory_size,
                     uint32_t t_cost, uint32_t m_cost, uint32_t p)
{
    static const unsigned char PASSWORD[BENCH_OUTLEN] = { 0 };
    static const unsigned char SALT[BENCH_INLEN] = { 1 };

    unsigned char out[BENCH_OUTLEN];
    struct timestamp start, end;
    double ms_d[BENCH_MAX_SAMPLES];
    double ms_i[BENCH_MAX_SAMPLES];
    double ms_id[BENCH_MAX_SAMPLES];

    double ms_d_final, ms_i_final, ms_id_final;
    unsigned int i, bench_samples;
    argon2_context ctx;

    int res;

    ctx.out = out;
    ctx.outlen = sizeof(out);
    ctx.pwd = (uint8_t *)PASSWORD;
    ctx.pwdlen = sizeof(PASSWORD);
    ctx.salt = (uint8_t *)SALT;
    ctx.saltlen = sizeof(SALT);
    ctx.secret = NULL;
    ctx.secretlen = 0;
    ctx.ad = NULL;
    ctx.adlen = 0;
    ctx.t_cost = t_cost;
    ctx.m_cost = m_cost;
    ctx.lanes = ctx.threads = p;
    ctx.version = ARGON2_VERSION_NUMBER;
    ctx.allocate_cbk = NULL;
    ctx.free_cbk = NULL;
    ctx.flags = ARGON2_DEFAULT_FLAGS;

    bench_samples = (BENCH_MIN_PASSES * p) / (t_cost * m_cost);
    bench_samples += (BENCH_MIN_PASSES * p) % (t_cost * m_cost) != 0;

    if (bench_samples > BENCH_MAX_SAMPLES) {
        bench_samples = BENCH_MAX_SAMPLES;
    }
    for (i = 0; i < bench_samples; i++) {
        timestamp_store(&start);
        res = argon2_ctx_mem(&ctx, Argon2_d, memory, memory_size);
        timestamp_store(&end);
        if (res != ARGON2_OK) {
            return res;
        }

        ms_d[i] = timestamp_span_ms(&start, &end);
    }

    for (i = 0; i < bench_samples; i++) {
        timestamp_store(&start);
        res = argon2_ctx_mem(&ctx, Argon2_i, memory, memory_size);
        timestamp_store(&end);
        if (res != ARGON2_OK) {
            return res;
        }

        ms_i[i] = timestamp_span_ms(&start, &end);
    }

    for (i = 0; i < bench_samples; i++) {
        timestamp_store(&start);
        res = argon2_ctx_mem(&ctx, Argon2_id, memory, memory_size);
        timestamp_store(&end);
        if (res != ARGON2_OK) {
            return res;
        }

        ms_id[i] = timestamp_span_ms(&start, &end);
    }

    ms_d_final = pick_min(ms_d, bench_samples);
    ms_i_final = pick_min(ms_i, bench_samples);
    ms_id_final = pick_min(ms_id, bench_samples);

    printf("%8lu%16lu%8lu%16.6lf%16.6lf%16.6lf\n",
           (unsigned long)t_cost, (unsigned long)m_cost, (unsigned long)p,
           ms_d_final, ms_i_final, ms_id_final);
    return 0;
}

int main(int argc, const char * const *argv)
{
    uint32_t max_t_cost = BENCH_MAX_T_COST;
    uint32_t max_m_cost = BENCH_MAX_M_COST;
    uint32_t max_p = BENCH_MAX_THREADS;
    uint32_t t_cost, m_cost, p;
    char *end;
    int res;

    if (argc >= 2) {
        max_t_cost = strtoul(argv[1], &end, 10);
        if (end == argv[1]) {
            fprintf(stderr, "ERROR: Invalid number format!\n");
            return 1;
        }
    }

    if (argc >= 3) {
        max_m_cost = strtoul(argv[2], &end, 10);
        if (end == argv[2]) {
            fprintf(stderr, "ERROR: Invalid number format!\n");
            return 1;
        }
    }

    if (argc >= 4) {
        max_p = strtoul(argv[3], &end, 10);
        if (end == argv[3]) {
            fprintf(stderr, "ERROR: Invalid number format!\n");
            return 1;
        }
    }

    argon2_select_impl(stderr, "[libargon2] ");

    size_t memory_size = (size_t)max_m_cost * (size_t)ARGON2_BLOCK_SIZE;
    void *memory = malloc(memory_size);
    if (memory == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failed!\n");
        return 1;
    }
    /* make sure the whole memory gets mapped to physical pages: */
    memset(memory, 0xAB, memory_size);

    printf("%8s%16s%8s%16s%16s%16s\n", "t_cost", "m_cost", "threads",
           "Argon2d (ms)", "Argon2i (ms)", "Argon2id (ms)");
    for (t_cost = 1; t_cost <= max_t_cost; t_cost *= 2) {
        uint32_t min_m_cost = max_p * ARGON2_SYNC_POINTS * 2;
        for (m_cost = min_m_cost; m_cost <= max_m_cost; m_cost *= 2) {
            for (p = 1; p <= max_p; p *= 2) {
                res = benchmark(memory, memory_size, t_cost, m_cost, p);
                if (res != 0) {
                    free(memory);
                    return res;
                }
            }
        }
    }
    free(memory);
    return 0;
}
