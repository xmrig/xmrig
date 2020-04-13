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

#ifndef XMRIG_CUDACONFIG_H
#define XMRIG_CUDACONFIG_H


#include "backend/cuda/CudaLaunchData.h"
#include "backend/common/Threads.h"
#include "backend/cuda/CudaThreads.h"


namespace xmrig {


class CudaConfig
{
public:
    CudaConfig() = default;

    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    std::vector<CudaLaunchData> get(const Miner *miner, const Algorithm &algorithm, const std::vector<CudaDevice> &devices) const;
    void read(const rapidjson::Value &value);

    inline bool isEnabled() const                               { return m_enabled; }
    inline bool isShouldSave() const                            { return m_shouldSave; }
    inline const std::vector<uint32_t> &devicesHint() const     { return m_devicesHint; }
    inline const String &loader() const                         { return m_loader; }
    inline const Threads<CudaThreads> &threads() const          { return m_threads; }
    inline int32_t bfactor() const                              { return m_bfactor; }
    inline int32_t bsleep() const                               { return m_bsleep; }

#   ifdef XMRIG_FEATURE_NVML
    inline bool isNvmlEnabled() const                           { return m_nvml; }
    inline const String &nvmlLoader() const                     { return m_nvmlLoader; }
#   endif

private:
    void generate();
    void setDevicesHint(const char *devicesHint);

    bool m_enabled          = false;
    bool m_shouldSave       = false;
    std::vector<uint32_t> m_devicesHint;
    String m_loader;
    Threads<CudaThreads> m_threads;

#   ifdef _WIN32
    int32_t m_bfactor      = 6;
    int32_t m_bsleep       = 25;
#   else
    int32_t m_bfactor      = 0;
    int32_t m_bsleep       = 0;
#   endif

#   ifdef XMRIG_FEATURE_NVML
    bool m_nvml            = true;
    String m_nvmlLoader;
#   endif
};


} /* namespace xmrig */


#endif /* XMRIG_CUDACONFIG_H */
