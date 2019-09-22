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

#ifndef XMRIG_OCLCONFIG_H
#define XMRIG_OCLCONFIG_H


#include "backend/common/Threads.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/OclThreads.h"
#include "backend/opencl/wrappers/OclPlatform.h"


namespace xmrig {


class OclConfig
{
public:
    OclConfig();

    OclPlatform platform() const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    std::vector<OclLaunchData> get(const Miner *miner, const Algorithm &algorithm, const OclPlatform &platform, const std::vector<OclDevice> &devices, const char *tag) const;
    void read(const rapidjson::Value &value);

    inline bool isCacheEnabled() const                  { return m_cache; }
    inline bool isEnabled() const                       { return m_enabled; }
    inline bool isShouldSave() const                    { return m_shouldSave; }
    inline const String &loader() const                 { return m_loader; }
    inline const Threads<OclThreads> &threads() const   { return m_threads; }

private:
    void generate();
    void setPlatform(const rapidjson::Value &platform);

    bool m_cache         = true;
    bool m_enabled       = false;
    bool m_shouldSave    = false;
    String m_loader;
    String m_platformVendor;
    Threads<OclThreads> m_threads;
    uint32_t m_platformIndex = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_OCLCONFIG_H */
