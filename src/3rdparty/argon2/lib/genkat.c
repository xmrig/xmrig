/*
 * Argon2 source code package
 *
 * Written by Daniel Dinu and Dmitry Khovratovich, 2015
 *
 * This work is licensed under a Creative Commons CC0 1.0 License/Waiver.
 *
 * You should have received a copy of the CC0 Public Domain Dedication along
 * with
 * this software. If not, see
 * <http://creativecommons.org/publicdomain/zero/1.0/>.
 */

#include <inttypes.h>
#include <stdio.h>

#include "genkat.h"

void initial_kat(const uint8_t *blockhash, const argon2_context *context,
                 argon2_type type) {
    unsigned i;

    if (blockhash != NULL && context != NULL) {
        printf("=======================================\n");

        printf("%s version number %d\n", argon2_type2string(type, 1),
               context->version);

        printf("=======================================\n");


        printf("Memory: %u KiB, Iterations: %u, Parallelism: %u lanes, Tag "
               "length: %u bytes\n",
               context->m_cost, context->t_cost, context->lanes,
               context->outlen);

        printf("Password[%u]: ", context->pwdlen);

        if (context->flags & ARGON2_FLAG_CLEAR_PASSWORD) {
            printf("CLEARED\n");
        } else {
            for (i = 0; i < context->pwdlen; ++i) {
                printf("%2.2x ", ((unsigned char *)context->pwd)[i]);
            }

            printf("\n");
        }

        printf("Salt[%u]: ", context->saltlen);

        for (i = 0; i < context->saltlen; ++i) {
            printf("%2.2x ", ((unsigned char *)context->salt)[i]);
        }

        printf("\n");

        printf("Secret[%u]: ", context->secretlen);

        if (context->flags & ARGON2_FLAG_CLEAR_SECRET) {
            printf("CLEARED\n");
        } else {
            for (i = 0; i < context->secretlen; ++i) {
                printf("%2.2x ", ((unsigned char *)context->secret)[i]);
            }

            printf("\n");
        }

        printf("Associated data[%u]: ", context->adlen);

        for (i = 0; i < context->adlen; ++i) {
            printf("%2.2x ", ((unsigned char *)context->ad)[i]);
        }

        printf("\n");

        printf("Pre-hashing digest: ");

        for (i = 0; i < ARGON2_PREHASH_DIGEST_LENGTH; ++i) {
            printf("%2.2x ", ((unsigned char *)blockhash)[i]);
        }

        printf("\n");
    }
}

void print_tag(const void *out, uint32_t outlen) {
    unsigned i;
    if (out != NULL) {
        printf("Tag: ");

        for (i = 0; i < outlen; ++i) {
            printf("%2.2x ", ((uint8_t *)out)[i]);
        }

        printf("\n");
    }
}

void internal_kat(const argon2_instance_t *instance, uint32_t pass) {

    if (instance != NULL) {
        uint32_t i, j;
        printf("\n After pass %u:\n", pass);

        for (i = 0; i < instance->memory_blocks; ++i) {
            uint32_t how_many_words =
                (instance->memory_blocks > ARGON2_QWORDS_IN_BLOCK)
                    ? 1
                    : ARGON2_QWORDS_IN_BLOCK;

            for (j = 0; j < how_many_words; ++j)
                printf("Block %.4u [%3u]: %016" PRIx64 "\n", i, j,
                       instance->memory[i].v[j]);
        }
    }
}
