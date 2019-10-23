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

#ifndef XMRIG_CUDACONFIG_H
#define XMRIG_CUDACONFIG_H


#include "backend/common/Threads.h"
#include "backend/cuda/CudaThreads.h"


namespace xmrig {


class CudaConfig
{
public:
    CudaConfig() = default;

    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    void read(const rapidjson::Value &value);

    inline bool isEnabled() const                       { return m_enabled; }
    inline bool isShouldSave() const                    { return m_shouldSave; }
    inline const Threads<CudaThreads> &threads() const  { return m_threads; }

private:
    void generate();
    void setDevicesHint(const char *devicesHint);

    bool m_enabled       = false;
    bool m_shouldSave    = false;
    std::vector<uint32_t> m_devicesHint;
    Threads<CudaThreads> m_threads;
};


} /* namespace xmrig */


#endif /* XMRIG_CUDACONFIG_H */
