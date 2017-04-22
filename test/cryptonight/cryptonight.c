#include <unity.h>
#include <stdbool.h>
#include <stdlib.h>
#include <algo/cryptonight/cryptonight.h>
#include <string.h>

const static char input1[76] = {
    0x03, 0x05, 0xA0, 0xDB, 0xD6, 0xBF, 0x05, 0xCF, 0x16, 0xE5, 0x03, 0xF3, 0xA6, 0x6F, 0x78, 0x00, 0x7C, 0xBF, 0x34,
    0x14, 0x43, 0x32, 0xEC, 0xBF, 0xC2, 0x2E, 0xD9, 0x5C, 0x87, 0x00, 0x38, 0x3B, 0x30, 0x9A, 0xCE, 0x19, 0x23, 0xA0,
    0x96, 0x4B, 0x00, 0x00, 0x00, 0x08, 0xBA, 0x93, 0x9A, 0x62, 0x72, 0x4C, 0x0D, 0x75, 0x81, 0xFC, 0xE5, 0x76, 0x1E,
    0x9D, 0x8A, 0x0E, 0x6A, 0x1C, 0x3F, 0x92, 0x4F, 0xDD, 0x84, 0x93, 0xD1, 0x11, 0x56, 0x49, 0xC0, 0x5E, 0xB6, 0x01
};

const static char input2[] = "This is a test";
const static char input3[] = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus pellentesque metus.";


void cryptonight_av1_aesni(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_av2_aesni_stak(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_av3_aesni_bmi2(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_av4_softaes(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);
void cryptonight_av5_aesni_experimental(const void* input, size_t size, void* output, struct cryptonight_ctx* ctx);


static char hash[32];
#define RESULT1 "1a3ffbee909b420d91f7be6e5fb56db71b3110d886011e877ee5786afd080100"
#define RESULT2 "a084f01d1437a09c6985401b60d43554ae105802c5f5d8a9b3253649c0be6605"
#define RESULT3 "0bbe54bd26caa92a1d436eec71cbef02560062fa689fe14d7efcf42566b411cf"


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

    cryptonight_av1_aesni(input1, sizeof(input1), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT1, bin2hex(hash, 32));

    cryptonight_av1_aesni(input2, strlen(input2), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT2, bin2hex(hash, 32));

    cryptonight_av1_aesni(input3, strlen(input3), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT3, bin2hex(hash, 32));

    free_ctx(ctx);
}


void test_cryptonight_av2_should_CalcHash(void)
{
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) create_ctx();

    cryptonight_av2_aesni_stak(input1, sizeof(input1), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT1, bin2hex(hash, 32));

    cryptonight_av2_aesni_stak(input2, strlen(input2), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT2, bin2hex(hash, 32));

    cryptonight_av2_aesni_stak(input3, strlen(input3), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT3, bin2hex(hash, 32));

    free_ctx(ctx);
}


void test_cryptonight_av3_should_CalcHash(void)
{
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) create_ctx();

    cryptonight_av3_aesni_bmi2(input1, sizeof(input1), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT1, bin2hex(hash, 32));

    cryptonight_av3_aesni_bmi2(input2, strlen(input2), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT2, bin2hex(hash, 32));

    cryptonight_av3_aesni_bmi2(input3, strlen(input3), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT3, bin2hex(hash, 32));

    free_ctx(ctx);
}


void test_cryptonight_av4_should_CalcHash(void)
{
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) create_ctx();

    cryptonight_av4_softaes(input1, sizeof(input1), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT1, bin2hex(hash, 32));

    cryptonight_av4_softaes(input2, strlen(input2), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT2, bin2hex(hash, 32));

    cryptonight_av4_softaes(input3, strlen(input3), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT3, bin2hex(hash, 32));

    free_ctx(ctx);
}


void test_cryptonight_av5_should_CalcHash(void)
{
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*) create_ctx();

    cryptonight_av5_aesni_experimental(input1, sizeof(input1), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT1, bin2hex(hash, 32));

    cryptonight_av5_aesni_experimental(input2, strlen(input2), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT2, bin2hex(hash, 32));

    cryptonight_av5_aesni_experimental(input3, strlen(input3), &hash, ctx);
    TEST_ASSERT_EQUAL_STRING(RESULT3, bin2hex(hash, 32));

    free_ctx(ctx);
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
