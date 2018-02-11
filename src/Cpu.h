#ifndef __CPU_H__
#define __CPU_H__
#include <stdint.h>
class Cpu
{
public:
    enum Flags {
        X86_64 = 1,
        AES    = 2,
        BMI2   = 4
    };
    static int optimalThreadsCount();
	static int CPUs();
    static void init();
    static void setAffinity(int id, uint64_t mask);
    static inline bool hasAES()       { return (m_flags & AES) != 0; }
    static inline bool isX64()        { return (m_flags & X86_64) != 0; }
    static inline const char *brand() { return m_brand; }
    static inline int cores()         { return m_totalCores; }
    static inline int l2()            { return m_l2_cache; }
    static inline int l3()            { return m_l3_cache; }
    static inline int sockets()       { return m_sockets; }
    static inline int threads()       { return m_totalThreads; }

private:
    static void initCommon();

    static bool m_l2_exclusive;
    static char m_brand[64];
    static int m_flags;
    static int m_l2_cache;
    static int m_l3_cache;
    static int m_sockets;
    static int m_totalCores;
    static int m_totalThreads;
};


#endif