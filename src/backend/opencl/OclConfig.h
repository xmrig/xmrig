/* xmlcore
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2020 xmlcore       <https://github.com/xmlcore>, <support@xmlcore.com>
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

#ifndef xmlcore_OCLCONFIG_H
#define xmlcore_OCLCONFIG_H


#include "backend/common/Threads.h"
#include "backend/opencl/OclLaunchData.h"
#include "backend/opencl/OclThreads.h"
#include "backend/opencl/wrappers/OclPlatform.h"


namespace xmlcore {


class OclConfig
{
public:
    OclConfig();

    OclPlatform platform() const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    std::vector<OclLaunchData> get(const Miner *miner, const Algorithm &algorithm, const OclPlatform &platform, const std::vector<OclDevice> &devices) const;
    void read(const rapidjson::Value &value);

    inline bool isCacheEnabled() const                  { return m_cache; }
    inline bool isEnabled() const                       { return m_enabled; }
    inline bool isShouldSave() const                    { return m_shouldSave; }
    inline const String &loader() const                 { return m_loader; }
    inline const Threads<OclThreads> &threads() const   { return m_threads; }

#   ifdef xmlcore_FEATURE_ADL
    inline bool isAdlEnabled() const                    { return m_adl; }
#   endif

private:
    void generate();
    void setDevicesHint(const char *devicesHint);

    bool m_cache         = true;
    bool m_enabled       = false;
    bool m_shouldSave    = false;
    std::vector<uint32_t> m_devicesHint;
    String m_loader;
    Threads<OclThreads> m_threads;

#   ifndef xmlcore_OS_APPLE
    void setPlatform(const rapidjson::Value &platform);

    String m_platformVendor;
    uint32_t m_platformIndex = 0;
#   endif

#   ifdef xmlcore_FEATURE_ADL
    bool m_adl          = true;
#   endif
};


} /* namespace xmlcore */


#endif /* xmlcore_OCLCONFIG_H */
