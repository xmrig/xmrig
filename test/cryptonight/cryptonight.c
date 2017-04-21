#include <unity.h>
#include <stdbool.h>
#include <stdlib.h>
#include <algo/cryptonight/cryptonight.h>

const static char input[76] = {
    0x03, 0x05, 0xA0, 0xDB, 0xD6, 0xBF, 0x05, 0xCF, 0x16, 0xE5, 0x03, 0xF3, 0xA6, 0x6F, 0x78, 0x00, 0x7C, 0xBF, 0x34,
    0x14, 0x43, 0x32, 0xEC, 0xBF, 0xC2, 0x2E, 0xD9, 0x5C, 0x87, 0x00, 0x38, 0x3B, 0x30, 0x9A, 0xCE, 0x19, 0x23, 0xA0,
    0x96, 0x4B, 0x00, 0x00, 0x00, 0x08, 0xBA, 0x93, 0x9A, 0x62, 0x72, 0x4C, 0x0D, 0x75, 0x81, 0xFC, 0xE5, 0x76, 0x1E,
    0x9D, 0x8A, 0x0E, 0x6A, 0x1C, 0x3F, 0x92, 0x4F, 0xDD, 0x84, 0x93, 0xD1, 0x11, 0x56, 0x49, 0xC0, 0x5E, 0xB6, 0x01
};


void cryptonight_av1_aesni(void* output, const void* input, struct cryptonight_ctx* ctx);
void cryptonight_av2_aesni_stak(void* output, const void* input, struct cryptonight_ctx* ctx);
void cryptonight_av3_aesni_bmi2(void* output, const void* input, struct cryptonight_ctx* ctx);
void cryptonight_av4_softaes(void* output, const void* input, struct cryptonight_ctx* ctx);
void cryptonight_av5_aesni_experimental(void* output, const void* input, struct cryptonight_ctx* ctx);


static char hash[32];
#define RESULT "1a3ffbee909b420d91f7be6e5fb56db71b3110d886011e877ee5786afd080100"


static char *bin2hex(const unsigned char *p, size_t len)
{
    char *s = malloc((len * 2) + 1);
    if (!s) {
        return NULL;
    }

    for (int i = 0; i < len; i++) {
        sprintf(s + (i * 2), "%02x", (unsigned int) p[i]);
    }

    return s;
}


static void * create_ctx() {
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) malloc(sizeof(struct cryptonight_ctx));
    ctx->memory = (uint8_t *) malloc(MEMORY);

    return ctx;
}


static void free_ctx(struct cryptonight_ctx *ctx) {
    free(ctx->memory);
    free(ctx);
}


void test_cryptonight_av1_should_CalcHash(void) {
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) create_ctx();

    cryptonight_av1_aesni(&hash, input, ctx);

    free_ctx(ctx);

    TEST_ASSERT_EQUAL_STRING(RESULT, bin2hex(hash, 32));
}


void test_cryptonight_av2_should_CalcHash(void)
{
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) create_ctx();

    cryptonight_av2_aesni_stak(&hash, input, ctx);

    free_ctx(ctx);

    TEST_ASSERT_EQUAL_STRING(RESULT, bin2hex(hash, 32));
}


void test_cryptonight_av3_should_CalcHash(void)
{
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) create_ctx();

    cryptonight_av3_aesni_bmi2(&hash, input, ctx);

    free_ctx(ctx);

    TEST_ASSERT_EQUAL_STRING(RESULT, bin2hex(hash, 32));
}


void test_cryptonight_av4_should_CalcHash(void)
{
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) create_ctx();

    cryptonight_av4_softaes(&hash, input, ctx);

    free_ctx(ctx);

    TEST_ASSERT_EQUAL_STRING(RESULT, bin2hex(hash, 32));
}


void test_cryptonight_av5_should_CalcHash(void)
{
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) create_ctx();

    cryptonight_av5_aesni_experimental(&hash, input, ctx);

    free_ctx(ctx);

    TEST_ASSERT_EQUAL_STRING(RESULT, bin2hex(hash, 32));
}


int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_cryptonight_av1_should_CalcHash);
    RUN_TEST(test_cryptonight_av2_should_CalcHash);
    RUN_TEST(test_cryptonight_av3_should_CalcHash);
    RUN_TEST(test_cryptonight_av4_should_CalcHash);
    RUN_TEST(test_cryptonight_av5_should_CalcHash);

    return UNITY_END();
}
