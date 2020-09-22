/* XMRig
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_PROFILER_H
#define XMRIG_PROFILER_H


#ifndef FORCE_INLINE
#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(__clang__)
#define FORCE_INLINE __inline__
#else
#define FORCE_INLINE
#endif
#endif


#ifdef XMRIG_FEATURE_PROFILING


#include <cstdint>
#include <cstddef>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif


static FORCE_INLINE uint64_t ReadTSC()
{
#ifdef _MSC_VER
    return __rdtsc();
#else
    uint32_t hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (((uint64_t)hi) << 32) | lo;
#endif
}


struct ProfileScopeData
{
    const char* m_name;
    uint64_t m_totalCycles;
    uint32_t m_totalSamples;

    enum
    {
        MAX_THREAD_ID_LENGTH = 11,
        MAX_SAMPLE_COUNT = 128,
        MAX_DATA_COUNT = 1024
    };

    char m_threadId[MAX_THREAD_ID_LENGTH + 1];

    static ProfileScopeData* s_data[MAX_DATA_COUNT];
    static volatile long s_dataCount;
    static double s_tscSpeed;

    static void Register(ProfileScopeData* data);
    static void Init();
};

static_assert(std::is_trivial<ProfileScopeData>::value, "ProfileScopeData must be a trivial struct");
static_assert(sizeof(ProfileScopeData) <= 32, "ProfileScopeData struct is too big");


class ProfileScope
{
public:
    FORCE_INLINE ProfileScope(ProfileScopeData& data)
        : m_data(data)
    {
        if (m_data.m_totalCycles == 0) {
            ProfileScopeData::Register(&data);
        }

        m_startCounter = ReadTSC();
    }

    FORCE_INLINE ~ProfileScope()
    {
        m_data.m_totalCycles += ReadTSC() - m_startCounter;
        ++m_data.m_totalSamples;
    }

private:
    ProfileScopeData& m_data;
    uint64_t m_startCounter;
};


#define PROFILE_SCOPE(x) static thread_local ProfileScopeData x##_data{#x}; ProfileScope x(x##_data);


#else /* XMRIG_FEATURE_PROFILING */
#define PROFILE_SCOPE(x)
#endif /* XMRIG_FEATURE_PROFILING */


#include "crypto/randomx/blake2/blake2.h"


struct rx_blake2b_wrapper
{
    FORCE_INLINE static void run(void* out, size_t outlen, const void* in, size_t inlen)
    {
        PROFILE_SCOPE(RandomX_Blake2b);
        rx_blake2b(out, outlen, in, inlen);
    }
};


#endif /* XMRIG_PROFILER_H */
