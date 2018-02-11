#ifndef __MEM_H__
#define __MEM_H__
#include <stddef.h>
#include <stdint.h>
#include "align.h"
struct cryptonight_ctx;
class Mem
{
public:
    enum Flags {
        HugepagesAvailable = 1,
        HugepagesEnabled   = 2,
        Lock               = 4
    };

    static bool allocate(int algo, int threads, bool doubleHash, bool enabled);
    static cryptonight_ctx *create(int threadId);
    static void *calloc(size_t num, size_t size);
    static void release();

    static inline bool isDoubleHash()         { return m_doubleHash; }
    static inline bool isHugepagesAvailable() { return (m_flags & HugepagesAvailable) != 0; }
    static inline bool isHugepagesEnabled()   { return (m_flags & HugepagesEnabled) != 0; }
    static inline int flags()                 { return m_flags; }
    static inline int threads()               { return m_threads; }

private:
    static bool m_doubleHash;
    static int m_algo;
    static int m_flags;
    static int m_threads;
    static size_t m_offset;
    VAR_ALIGN(16, static uint8_t *m_memory);
};


#endif /* __MEM_H__ */
