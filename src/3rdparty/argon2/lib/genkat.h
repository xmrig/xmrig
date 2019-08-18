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

#ifndef ARGON2_KAT_H
#define ARGON2_KAT_H

#include "core.h"

/*
 * Initial KAT function that prints the inputs to the file
 * @param  blockhash  Array that contains pre-hashing digest
 * @param  context Holds inputs
 * @param  type Argon2 type
 * @pre blockhash must point to INPUT_INITIAL_HASH_LENGTH bytes
 * @pre context member pointers must point to allocated memory of size according
 * to the length values
 */
void initial_kat(const uint8_t *blockhash, const argon2_context *context,
                 argon2_type type);

/*
 * Function that prints the output tag
 * @param  out  output array pointer
 * @param  outlen digest length
 * @pre out must point to @a outlen bytes
 **/
void print_tag(const void *out, uint32_t outlen);

/*
 * Function that prints the internal state at given moment
 * @param  instance pointer to the current instance
 * @param  pass current pass number
 * @pre instance must have necessary memory allocated
 **/
void internal_kat(const argon2_instance_t *instance, uint32_t pass);

#endif
