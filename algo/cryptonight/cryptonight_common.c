/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2016-2017 XMRig       <support@xmrig.com>
 *
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include <stdlib.h>

#ifndef BUILD_TEST
# include "xmrig.h"
#endif

#include "crypto/c_groestl.h"
#include "crypto/c_blake256.h"
#include "crypto/c_jh.h"
#include "crypto/c_skein.h"
#include "cryptonight.h"
#include "options.h"


#if defined(__x86_64__)
  void cryptonight_av1_aesni(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
  void cryptonight_av2_aesni_stak(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
  void cryptonight_av3_aesni_bmi2(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
  void cryptonight_av4_softaes(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
  void cryptonight_av5_aesni_experimental(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
#elif defined(__i386__)
  void cryptonight_av1_aesni32(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);
#endif

void cryptonight_av4_softaes(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx);

void (*cryptonight_hash_ctx)(void* output, const void* input, const char *memory, struct cryptonight_ctx* ctx) = NULL;


void cryptonight_init(int variant)
{
    switch (variant) {
        #if defined(__x86_64__)
        case XMR_AV1_AESNI:
            cryptonight_hash_ctx = cryptonight_av1_aesni;
            break;

        case XMR_AV2_STAK:
            cryptonight_hash_ctx = cryptonight_av2_aesni_stak;
            break;

        case XMR_AV3_AESNI_BMI2:
            cryptonight_hash_ctx = cryptonight_av3_aesni_bmi2;
            break;

        case XMR_AV5_EXPERIMENTAL:
            cryptonight_hash_ctx = cryptonight_av5_aesni_experimental;
            break;

        #elif defined(__i386__)
        case XMR_VARIANT_AESNI:
            cryptonight_hash_ctx = cryptonight_av1_aesni32;
            break;
        #endif

        case XMR_AV4_SOFT_AES:
             cryptonight_hash_ctx = cryptonight_av4_softaes;
             break;

        default:
            break;
    }

}


static inline void do_blake_hash(const void* input, size_t len, char* output) {
    blake256_hash((uint8_t*)output, input, len);
}


static inline void do_groestl_hash(const void* input, size_t len, char* output) {
    groestl(input, len * 8, (uint8_t*)output);
}


static inline void do_jh_hash(const void* input, size_t len, char* output) {
    jh_hash(32 * 8, input, 8 * len, (uint8_t*)output);
}


static inline void do_skein_hash(const void* input, size_t len, char* output) {
    skein_hash(8 * 32, input, 8 * len, (uint8_t*)output);
}


void (* const extra_hashes[4])(const void *, size_t, char *) = {do_blake_hash, do_groestl_hash, do_jh_hash, do_skein_hash};


void cryptonight_hash(void* output, const void* input, size_t len) {
    uint8_t *memory __attribute((aligned(16))) = (uint8_t *) malloc(MEMORY);
    struct cryptonight_ctx *ctx = (struct cryptonight_ctx*)malloc(sizeof(struct cryptonight_ctx));

    cryptonight_hash_ctx(output, input, memory, ctx);

    free(memory);
    free(ctx);
}


#ifndef BUILD_TEST
int scanhash_cryptonight(int thr_id, uint32_t *hash, uint32_t *restrict pdata, const uint32_t *restrict ptarget, uint32_t max_nonce, unsigned long *restrict hashes_done, const char *restrict memory, struct cryptonight_ctx *persistentctx) {
    uint32_t *nonceptr = (uint32_t*) (((char*)pdata) + 39);
    uint32_t n = *nonceptr - 1;
    const uint32_t first_nonce = n + 1;

    do {
        *nonceptr = ++n;
        cryptonight_hash_ctx(hash, pdata, memory, persistentctx);

        if (unlikely(hash[7] < ptarget[7])) {
            *hashes_done = n - first_nonce + 1;
            return true;
        }
    } while (likely((n <= max_nonce && !work_restart[thr_id].restart)));

    *hashes_done = n - first_nonce + 1;
    return 0;
}
#endif
