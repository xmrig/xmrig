/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CPUCONFIG_H
#define XMRIG_CPUCONFIG_H


#include "backend/common/Threads.h"
#include "backend/cpu/CpuLaunchData.h"
#include "backend/cpu/CpuThreads.h"
#include "crypto/common/Assembly.h"


namespace xmrig {


class CpuConfig
{
public:
    enum AesMode {
        AES_AUTO,
        AES_HW,
        AES_SOFT
    };

    static const char *kEnabled;
    static const char *kField;
    static const char *kHugePages;
    static const char *kHugePagesJit;
    static const char *kHwAes;
    static const char *kMaxThreadsHint;
    static const char *kMemoryPool;
    static const char *kPriority;
    static const char *kYield;

#   ifdef XMRIG_FEATURE_ASM
    static const char *kAsm;
#   endif

#   ifdef XMRIG_ALGO_ARGON2
    static const char *kArgon2Impl;
#   endif

    CpuConfig() = default;

    bool isHwAES() const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    size_t memPoolSize() const;
    std::vector<CpuLaunchData> get(const Miner *miner, const Algorithm &algorithm) const;
    void read(const rapidjson::Value &value);

    inline bool isEnabled() const                       { return m_enabled; }
    inline bool isHugePages() const                     { return m_hugePageSize > 0; }
    inline bool isHugePagesJit() const                  { return m_hugePagesJit; }
    inline bool isShouldSave() const                    { return m_shouldSave; }
    inline bool isYield() const                         { return m_yield; }
    inline const Assembly &assembly() const             { return m_assembly; }
    inline const String &argon2Impl() const             { return m_argon2Impl; }
    inline const Threads<CpuThreads> &threads() const   { return m_threads; }
    inline int priority() const                         { return m_priority; }
    inline size_t hugePageSize() const                  { return m_hugePageSize * 1024U; }
    inline uint32_t limit() const                       { return m_limit; }

private:
    constexpr static size_t kDefaultHugePageSizeKb  = 2048U;
    constexpr static size_t kOneGbPageSizeKb        = 1048576U;

    void generate();
    void setAesMode(const rapidjson::Value &value);
    void setHugePages(const rapidjson::Value &value);
    void setMemoryPool(const rapidjson::Value &value);

    inline void setPriority(int priority)   { m_priority = (priority >= -1 && priority <= 5) ? priority : -1; }

    AesMode m_aes           = AES_AUTO;
    Assembly m_assembly;
    bool m_enabled          = true;
    bool m_hugePagesJit     = false;
    bool m_shouldSave       = false;
    bool m_yield            = true;
    int m_memoryPool        = 0;
    int m_priority          = -1;
    size_t m_hugePageSize   = kDefaultHugePageSizeKb;
    String m_argon2Impl;
    Threads<CpuThreads> m_threads;
    uint32_t m_limit        = 100;
};


} /* namespace xmrig */


#endif /* XMRIG_CPUCONFIG_H */
