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

    static int optimalThreadsCount(int algo, bool doubleHash, int maxCpuUsage);
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


#endif /* __CPU_H__ */
