/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include "backend/cpu/CpuThread.h"
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

    CpuConfig();

    bool isHwAES() const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    std::vector<CpuLaunchData> get(const Miner *miner, const Algorithm &algorithm) const;
    void read(const rapidjson::Value &value);

    inline bool isEnabled() const                    { return m_enabled; }
    inline bool isHugePages() const                  { return m_hugePages; }
    inline bool isShouldSave() const                 { return m_shouldSave; }
    inline const Assembly &assembly() const          { return m_assembly; }
    inline const Threads<CpuThread> &threads() const { return m_threads; }
    inline int priority() const                      { return m_priority; }

private:
    void generate();
    void setAesMode(const rapidjson::Value &aesMode);

    inline void setPriority(int priority)   { m_priority = (priority >= -1 && priority <= 5) ? priority : -1; }

    AesMode m_aes        = AES_AUTO;
    Assembly m_assembly;
    bool m_enabled       = true;
    bool m_hugePages     = true;
    bool m_shouldSave    = false;
    int m_priority       = -1;
    Threads<CpuThread> m_threads;
};


} /* namespace xmrig */


#endif /* XMRIG_CPUCONFIG_H */
