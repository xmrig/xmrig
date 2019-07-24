#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "argon2.h"

static void fatal(const char *error) {
    fprintf(stderr, "Error: %s\n", error);
    exit(1);
}

static void generate_testvectors(argon2_type type, const uint32_t version) {
#define TEST_OUTLEN 32
#define TEST_PWDLEN 32
#define TEST_SALTLEN 16
#define TEST_SECRETLEN 8
#define TEST_ADLEN 12
    argon2_context context;

    unsigned char out[TEST_OUTLEN];
    unsigned char pwd[TEST_PWDLEN];
    unsigned char salt[TEST_SALTLEN];
    unsigned char secret[TEST_SECRETLEN];
    unsigned char ad[TEST_ADLEN];
    const allocate_fptr myown_allocator = NULL;
    const deallocate_fptr myown_deallocator = NULL;

    unsigned t_cost = 3;
    unsigned m_cost = 32;
    unsigned lanes = 4;

    memset(pwd, 1, TEST_OUTLEN);
    memset(salt, 2, TEST_SALTLEN);
    memset(secret, 3, TEST_SECRETLEN);
    memset(ad, 4, TEST_ADLEN);

    context.out = out;
    context.outlen = TEST_OUTLEN;
    context.version = version;
    context.pwd = pwd;
    context.pwdlen = TEST_PWDLEN;
    context.salt = salt;
    context.saltlen = TEST_SALTLEN;
    context.secret = secret;
    context.secretlen = TEST_SECRETLEN;
    context.ad = ad;
    context.adlen = TEST_ADLEN;
    context.t_cost = t_cost;
    context.m_cost = m_cost;
    context.lanes = lanes;
    context.threads = lanes;
    context.allocate_cbk = myown_allocator;
    context.free_cbk = myown_deallocator;
    context.flags = ARGON2_DEFAULT_FLAGS | ARGON2_FLAG_GENKAT;

#undef TEST_OUTLEN
#undef TEST_PWDLEN
#undef TEST_SALTLEN
#undef TEST_SECRETLEN
#undef TEST_ADLEN

    argon2_ctx(&context, type);
}

int main(int argc, char *argv[]) {
    /* Get and check Argon2 type */
    const char *type_str = (argc > 1) ? argv[1] : "i";
    argon2_type type = Argon2_i;
    uint32_t version = ARGON2_VERSION_NUMBER;
    if (!strcmp(type_str, "d")) {
        type = Argon2_d;
    } else if (!strcmp(type_str, "i")) {
        type = Argon2_i;
    } else if (!strcmp(type_str, "id")) {
        type = Argon2_id;
    } else {
        fatal("wrong Argon2 type");
    }

    /* Get and check Argon2 version number */
    if(argc > 2) {
        version = strtoul(argv[2], NULL, 10);
    }
    if (ARGON2_VERSION_10 != version && ARGON2_VERSION_NUMBER != version) {
        fatal("wrong Argon2 version number");
    }

    generate_testvectors(type, version);
    return ARGON2_OK;
}
