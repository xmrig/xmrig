#ifndef __CRYPTONIGHT_H__
#define __CRYPTONIGHT_H__


#include <stddef.h>
#include <stdint.h>


#include "align.h"


#define MEMORY      2097152 /* 2 MiB */
#define MEMORY_LITE 1048576 /* 1 MiB */


struct cryptonight_ctx {
    VAR_ALIGN(16, uint8_t state0[200]);
    VAR_ALIGN(16, uint8_t state1[200]);
    VAR_ALIGN(16, uint8_t* memory);
};


class Job;
class JobResult;


class CryptoNight
{
public:
    static bool hash(const Job &job, JobResult &result, cryptonight_ctx *ctx);
    static bool init(int algo, int variant);
    static void hash(const uint8_t *input, size_t size, uint8_t *output, cryptonight_ctx *ctx);

private:
    static bool selfTest(int algo);
};

#endif /* __CRYPTONIGHT_H__ */
