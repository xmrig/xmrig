#include "crypto/CryptoNight.h"
#include "crypto/CryptoNight_p.h"
#include "crypto/CryptoNight_test.h"
#include "net/Job.h"
#include "net/JobResult.h"
#include "Options.h"
void (*cryptonight_hash_ctx)(const void *input, size_t size, void *output, cryptonight_ctx *ctx) = nullptr;
static void cryptonight_av1_aesni(const void *input, size_t size, void *output, struct cryptonight_ctx *ctx) {
    cryptonight_hash<0x80000, MEMORY, 0x1FFFF0, false>(input, size, output, ctx);
}
static void cryptonight_av2_aesni_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_double_hash<0x80000, MEMORY, 0x1FFFF0, false>(input, size, output, ctx);
}
static void cryptonight_av3_softaes(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_hash<0x80000, MEMORY, 0x1FFFF0, true>(input, size, output, ctx);
}
static void cryptonight_av4_softaes_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_double_hash<0x80000, MEMORY, 0x1FFFF0, true>(input, size, output, ctx);
}
#ifndef XMRIG_NO_AEON
static void cryptonight_lite_av1_aesni(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_hash<0x40000, MEMORY_LITE, 0xFFFF0, false>(input, size, output, ctx);
}
static void cryptonight_lite_av2_aesni_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_double_hash<0x40000, MEMORY_LITE, 0xFFFF0, false>(input, size, output, ctx);
}
static void cryptonight_lite_av3_softaes(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_hash<0x40000, MEMORY_LITE, 0xFFFF0, true>(input, size, output, ctx);
}
static void cryptonight_lite_av4_softaes_double(const void *input, size_t size, void *output, cryptonight_ctx *ctx) {
    cryptonight_double_hash<0x40000, MEMORY_LITE, 0xFFFF0, true>(input, size, output, ctx);
}
void (*cryptonight_variations[8])(const void *input, size_t size, void *output, cryptonight_ctx *ctx) = {
            cryptonight_av1_aesni,
            cryptonight_av2_aesni_double,
            cryptonight_av3_softaes,
            cryptonight_av4_softaes_double,
            cryptonight_lite_av1_aesni,
            cryptonight_lite_av2_aesni_double,
            cryptonight_lite_av3_softaes,
            cryptonight_lite_av4_softaes_double
        };
#else
void (*cryptonight_variations[4])(const void *input, size_t size, void *output, cryptonight_ctx *ctx) = {
            cryptonight_av1_aesni,
            cryptonight_av2_aesni_double,
            cryptonight_av3_softaes,
            cryptonight_av4_softaes_double
        };
#endif
bool CryptoNight::hash(const Job &job, JobResult &result, cryptonight_ctx *ctx)
{
    cryptonight_hash_ctx(job.blob(), job.size(), result.result, ctx);

    return *reinterpret_cast<uint64_t*>(result.result + 24) < job.target();
}
bool CryptoNight::init(int algo, int variant)
{
    if (variant < 1 || variant > 4) {
        return false;
    }

#   ifndef XMRIG_NO_AEON
    const int index = algo == Options::ALGO_CRYPTONIGHT_LITE ? (variant + 3) : (variant - 1);
#   else
    const int index = variant - 1;
#   endif
    cryptonight_hash_ctx = cryptonight_variations[index];
    return selfTest(algo);
}
void CryptoNight::hash(const uint8_t *input, size_t size, uint8_t *output, cryptonight_ctx *ctx)
{
    cryptonight_hash_ctx(input, size, output, ctx);
}
bool CryptoNight::selfTest(int algo) {
    if (cryptonight_hash_ctx == nullptr) {
        return false;
    }
    char output[64];
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) _mm_malloc(sizeof(struct cryptonight_ctx), 16);
    ctx->memory = (uint8_t *) _mm_malloc(MEMORY * 2, 16);
    cryptonight_hash_ctx(test_input, 76, output, ctx);
    _mm_free(ctx->memory);
    _mm_free(ctx);

#   ifndef XMRIG_NO_AEON
    return memcmp(output, algo == Options::ALGO_CRYPTONIGHT_LITE ? test_output1 : test_output0, (Options::i()->doubleHash() ? 64 : 32)) == 0;
#   else
    return memcmp(output, test_output0, (Options::i()->doubleHash() ? 64 : 32)) == 0;
#   endif
}
