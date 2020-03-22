/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

    CpuConfig() = default;

    bool isHwAES() const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    size_t memPoolSize() const;
    std::vector<CpuLaunchData> get(const Miner *miner, const Algorithm &algorithm) const;
    void read(const rapidjson::Value &value);

    inline bool isEnabled() const                       { return m_enabled; }
    inline bool isHugePages() const                     { return m_hugePages; }
    inline bool isShouldSave() const                    { return m_shouldSave; }
    inline bool isYield() const                         { return m_yield; }
    inline const Assembly &assembly() const             { return m_assembly; }
    inline const String &argon2Impl() const             { return m_argon2Impl; }
    inline const Threads<CpuThreads> &threads() const   { return m_threads; }
    inline int astrobwtMaxSize() const                  { return m_astrobwtMaxSize; }
    inline bool astrobwtAVX2() const                    { return m_astrobwtAVX2; }
    inline int priority() const                         { return m_priority; }
    inline uint32_t limit() const                       { return m_limit; }

private:
    void generate();
    void setAesMode(const rapidjson::Value &value);
    void setMemoryPool(const rapidjson::Value &value);

    inline void setPriority(int priority)   { m_priority = (priority >= -1 && priority <= 5) ? priority : -1; }

    AesMode m_aes           = AES_AUTO;
    Assembly m_assembly;
    bool m_astrobwtAVX2     = false;
    bool m_enabled          = true;
    bool m_hugePages        = true;
    bool m_shouldSave       = false;
    bool m_yield            = true;
    int m_astrobwtMaxSize   = 550;
    int m_memoryPool        = 0;
    int m_priority          = -1;
    String m_argon2Impl;
    Threads<CpuThreads> m_threads;
    uint32_t m_limit        = 100;
};


} /* namespace xmrig */


#endif /* XMRIG_CPUCONFIG_H */
