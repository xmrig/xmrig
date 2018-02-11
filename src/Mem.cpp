#include <memory.h>
#include "crypto/CryptoNight.h"
#include "Mem.h"
#include "Options.h"
bool Mem::m_doubleHash = false;
int Mem::m_algo        = 0;
int Mem::m_flags       = 0;
int Mem::m_threads     = 0;
size_t Mem::m_offset   = 0;
uint8_t *Mem::m_memory = nullptr;
cryptonight_ctx *Mem::create(int threadId)
{
    cryptonight_ctx *ctx = reinterpret_cast<cryptonight_ctx *>(&m_memory[MEMORY - sizeof(cryptonight_ctx) * (threadId + 1)]);
    const int ratio = m_doubleHash ? 2 : 1;
    ctx->memory = &m_memory[MEMORY * (threadId * ratio + 1)];
    return ctx;
}
void *Mem::calloc(size_t num, size_t size)
{
    void *mem = &m_memory[m_offset];
    m_offset += (num * size);
    memset(mem, 0, num * size);
    return mem;
}