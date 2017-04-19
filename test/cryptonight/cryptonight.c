#include <unity.h>
#include <stdbool.h>
#include <stdlib.h>
#include <algo/cryptonight/cryptonight.h>


void cryptonight_av1_aesni(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
void cryptonight_av2_aesni_wolf(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
void cryptonight_av3_aesni_bmi2(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
void cryptonight_av4_softaes(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
void cryptonight_av5_aesni_stak(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
void cryptonight_av6_aesni_experimental(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);


char *bin2hex(const unsigned char *p, size_t len)
{
    int i;
    char *s = malloc((len * 2) + 1);
    if (!s)
        return NULL;

    for (i = 0; i < len; i++)
        sprintf(s + (i * 2), "%02x", (unsigned int) p[i]);

    return s;
}

bool hex2bin(unsigned char *p, const char *hexstr, size_t len)
{
    char hex_byte[3];
    char *ep;

    hex_byte[2] = '\0';

    while (*hexstr && len) {
        if (!hexstr[1]) {
            return false;
        }
        hex_byte[0] = hexstr[0];
        hex_byte[1] = hexstr[1];
        *p = (unsigned char) strtol(hex_byte, &ep, 16);
        if (*ep) {
            return false;
        }
        p++;
        hexstr += 2;
        len--;
    }

    return (len == 0 && *hexstr == 0) ? true : false;
}


void test_cryptonight_av1_should_CalcHash(void) {
    char hash[32];
    char data[76];

    hex2bin((unsigned char *) &data, "0305a0dbd6bf05cf16e503f3a66f78007cbf34144332ecbfc22ed95c8700383b309ace1923a0964b00000008ba939a62724c0d7581fce5761e9d8a0e6a1c3f924fdd8493d1115649c05eb601", 76);

    uint8_t *memory = (uint8_t *) malloc(MEMORY);
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*)malloc(sizeof(struct cryptonight_ctx));

    cryptonight_av1_aesni(&hash, data, memory, ctx);

    free(memory);
    free(ctx);

    TEST_ASSERT_EQUAL_STRING("1a3ffbee909b420d91f7be6e5fb56db71b3110d886011e877ee5786afd080100", bin2hex(hash, 32));
}


void test_cryptonight_av2_should_CalcHash(void)
{
    char hash[32];
    char data[76];

    hex2bin((unsigned char *) &data, "0305a0dbd6bf05cf16e503f3a66f78007cbf34144332ecbfc22ed95c8700383b309ace1923a0964b00000008ba939a62724c0d7581fce5761e9d8a0e6a1c3f924fdd8493d1115649c05eb601", 76);

    uint8_t *memory = (uint8_t *) malloc(MEMORY);
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*)malloc(sizeof(struct cryptonight_ctx));

    cryptonight_av2_aesni_wolf(&hash, data, memory, ctx);

    free(memory);
    free(ctx);

    TEST_ASSERT_EQUAL_STRING("1a3ffbee909b420d91f7be6e5fb56db71b3110d886011e877ee5786afd080100", bin2hex(hash, 32));
}


void test_cryptonight_av3_should_CalcHash(void)
{
    char hash[32];
    char data[76];

    hex2bin((unsigned char *) &data, "0305a0dbd6bf05cf16e503f3a66f78007cbf34144332ecbfc22ed95c8700383b309ace1923a0964b00000008ba939a62724c0d7581fce5761e9d8a0e6a1c3f924fdd8493d1115649c05eb601", 76);

    uint8_t *memory = (uint8_t *) malloc(MEMORY);
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*)malloc(sizeof(struct cryptonight_ctx));

    cryptonight_av3_aesni_bmi2(&hash, data, memory, ctx);

    free(memory);
    free(ctx);

    TEST_ASSERT_EQUAL_STRING("1a3ffbee909b420d91f7be6e5fb56db71b3110d886011e877ee5786afd080100", bin2hex(hash, 32));
}


void test_cryptonight_av4_should_CalcHash(void)
{
    char hash[32];
    char data[76];

    hex2bin((unsigned char *) &data, "0305a0dbd6bf05cf16e503f3a66f78007cbf34144332ecbfc22ed95c8700383b309ace1923a0964b00000008ba939a62724c0d7581fce5761e9d8a0e6a1c3f924fdd8493d1115649c05eb601", 76);

    uint8_t *memory = (uint8_t *) malloc(MEMORY);
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*)malloc(sizeof(struct cryptonight_ctx));

    cryptonight_av4_softaes(&hash, data, memory, ctx);

    free(memory);
    free(ctx);

    TEST_ASSERT_EQUAL_STRING("1a3ffbee909b420d91f7be6e5fb56db71b3110d886011e877ee5786afd080100", bin2hex(hash, 32));
}


void test_cryptonight_av5_should_CalcHash(void)
{
    char hash[32];
    char data[76];

    hex2bin((unsigned char *) &data, "0305a0dbd6bf05cf16e503f3a66f78007cbf34144332ecbfc22ed95c8700383b309ace1923a0964b00000008ba939a62724c0d7581fce5761e9d8a0e6a1c3f924fdd8493d1115649c05eb601", 76);

    uint8_t *memory = (uint8_t *) malloc(MEMORY);
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*)malloc(sizeof(struct cryptonight_ctx));

    cryptonight_av5_aesni_stak(&hash, data, memory, ctx);

    free(memory);
    free(ctx);

    TEST_ASSERT_EQUAL_STRING("1a3ffbee909b420d91f7be6e5fb56db71b3110d886011e877ee5786afd080100", bin2hex(hash, 32));
}


void test_cryptonight_av6_should_CalcHash(void)
{
    char hash[32];
    char data[76];

    hex2bin((unsigned char *) &data, "0305a0dbd6bf05cf16e503f3a66f78007cbf34144332ecbfc22ed95c8700383b309ace1923a0964b00000008ba939a62724c0d7581fce5761e9d8a0e6a1c3f924fdd8493d1115649c05eb601", 76);

    uint8_t *memory = (uint8_t *) malloc(MEMORY);
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*)malloc(sizeof(struct cryptonight_ctx));

    cryptonight_av6_aesni_experimental(&hash, data, memory, ctx);

    free(memory);
    free(ctx);

    TEST_ASSERT_EQUAL_STRING("1a3ffbee909b420d91f7be6e5fb56db71b3110d886011e877ee5786afd080100", bin2hex(hash, 32));
}


int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_cryptonight_av1_should_CalcHash);
    RUN_TEST(test_cryptonight_av2_should_CalcHash);
    RUN_TEST(test_cryptonight_av3_should_CalcHash);
    RUN_TEST(test_cryptonight_av4_should_CalcHash);
    RUN_TEST(test_cryptonight_av5_should_CalcHash);
    RUN_TEST(test_cryptonight_av6_should_CalcHash);

    return UNITY_END();
}
