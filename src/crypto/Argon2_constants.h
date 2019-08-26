#ifndef XMRIG_ARGON2_CONSTANTS_H
#define XMRIG_ARGON2_CONSTANTS_H


#include <stddef.h>
#include <stdint.h>


#include "common/xmrig.h"

namespace xmrig
{
    enum Argon2Algo {
        I = 0,
        D = 1,
        ID = 2
    };

    constexpr const size_t    ARGON2_SALTLEN                  = 16;
    constexpr const size_t    ARGON2_HASHLEN                  = 32;

    constexpr const size_t    ARGON2_MEMORY_CHUKWA            = 512;
    constexpr const size_t    ARGON2_ITERS_CHUKWA             = 3;
    constexpr const size_t    ARGON2_PARALLELISM_CHUKWA       = 1;

    constexpr const size_t    ARGON2_MEMORY_CHUKWA_LITE       = 256;
    constexpr const size_t    ARGON2_ITERS_CHUKWA_LITE        = 4;
    constexpr const size_t    ARGON2_PARALLELISM_CHUKWA_LITE  = 1;

    constexpr const int       ARGON2_ALGO_CHUKWA              = Argon2Algo::ID;

    inline int argon2_select_algo(Variant variant)
    {
        switch (variant)
        {
            case VARIANT_CHUKWA:
                return ARGON2_ALGO_CHUKWA;
            case VARIANT_CHUKWA_LITE:
                return ARGON2_ALGO_CHUKWA;
        }

        return 0;
    }

    inline uint64_t argon2_select_memory(Variant variant)
    {
        switch (variant)
        {
            case VARIANT_CHUKWA:
                return ARGON2_MEMORY_CHUKWA;
            case VARIANT_CHUKWA_LITE:
                return ARGON2_MEMORY_CHUKWA_LITE;
        }

        return 0;
    }

    inline uint32_t argon2_select_iters(Variant variant)
    {
        switch (variant)
        {
            case VARIANT_CHUKWA:
                return ARGON2_ITERS_CHUKWA;
            case VARIANT_CHUKWA_LITE:
                return ARGON2_ITERS_CHUKWA_LITE;
        }

        return 0;
    }

    inline uint32_t argon2_select_parallelism(Variant variant)
    {
        switch (variant)
        {
            case VARIANT_CHUKWA:
                return ARGON2_PARALLELISM_CHUKWA;
            case VARIANT_CHUKWA_LITE:
                return ARGON2_PARALLELISM_CHUKWA_LITE;
        }

        return 0;
    }
}

#endif