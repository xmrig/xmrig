#ifndef XMRIG_ARGON2_H
#define XMRIG_ARGON2_H

#include <argon2.h>

#include "crypto/Argon2_constants.h"

static bool argon_optimization_selected = false;

template<xmrig::Variant VARIANT>
inline void argon2_hash_function(const uint8_t *__restrict__ input, size_t size, uint8_t *__restrict__ output, cryptonight_ctx **__restrict__ ctx, uint64_t height)
{
    /* If this is the first time we've called this hash function then
       we need to have the Argon2 library check to see if any of the
       available CPU instruction sets are going to help us out */
    if (!argon_optimization_selected)
    {
      /* Call the library quick benchmark test to set which CPU
         instruction sets will be used */
      argon2_select_impl(NULL, NULL);

      argon_optimization_selected = true;
    }

     uint8_t salt[xmrig::ARGON2_SALTLEN];

     memcpy(salt, input, sizeof(salt));

     const uint32_t ITERS         = xmrig::argon2_select_iters(VARIANT);
     const uint32_t MEMORY        = xmrig::argon2_select_memory(VARIANT);
     const uint32_t PARALLELISM   = xmrig::argon2_select_parallelism(VARIANT);
     const int      ALGO          = xmrig::argon2_select_algo(VARIANT);

     switch (ALGO)
     {
          case xmrig::Argon2Algo::I:
              argon2i_hash_raw(ITERS, MEMORY, PARALLELISM, input, size, salt, xmrig::ARGON2_SALTLEN, output, xmrig::ARGON2_HASHLEN);
          case xmrig::Argon2Algo::D:
              argon2d_hash_raw(ITERS, MEMORY, PARALLELISM, input, size, salt, xmrig::ARGON2_SALTLEN, output, xmrig::ARGON2_HASHLEN);
          case xmrig::Argon2Algo::ID:
              argon2id_hash_raw(ITERS, MEMORY, PARALLELISM, input, size, salt, xmrig::ARGON2_SALTLEN, output, xmrig::ARGON2_HASHLEN);
     }
}

#endif