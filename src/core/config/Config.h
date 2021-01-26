/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CONFIG_H
#define XMRIG_CONFIG_H


#include <cstdint>


#include "3rdparty/rapidjson/fwd.h"
#include "backend/cpu/CpuConfig.h"
#include "base/kernel/config/BaseConfig.h"
#include "base/tools/Object.h"
#ifdef XMRIG_FEATURE_MO_BENCHMARK
#include "core/MoBenchmark.h"
#endif


namespace xmrig {


class ConfigPrivate;
class CudaConfig;
class IThread;
class OclConfig;
class RxConfig;


class Config : public BaseConfig
{
public:
    XMRIG_DISABLE_COPY_MOVE(Config);

#   ifdef XMRIG_FEATURE_OPENCL
    static const char *kOcl;
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    static const char *kCuda;
#   endif

#   if defined(XMRIG_FEATURE_NVML) || defined (XMRIG_FEATURE_ADL)
    static const char *kHealthPrintTime;
#   endif

#   ifdef XMRIG_FEATURE_DMI
    static const char *kDMI;
#   endif

    Config();
    ~Config() override;

    const CpuConfig &cpu() const;

#   ifdef XMRIG_FEATURE_OPENCL
    const OclConfig &cl() const;
#   endif

#   ifdef XMRIG_FEATURE_CUDA
    const CudaConfig &cuda() const;
#   endif

#   ifdef XMRIG_ALGO_RANDOMX
    const RxConfig &rx() const;
#   endif

#   if defined(XMRIG_FEATURE_NVML) || defined (XMRIG_FEATURE_ADL)
    uint32_t healthPrintTime() const;
#   else
    uint32_t healthPrintTime() const        { return 0; }
#   endif

#   ifdef XMRIG_FEATURE_DMI
    bool isDMI() const;
#   else
    static constexpr inline bool isDMI()    { return false; }
#   endif

    bool isShouldSave() const;
    bool read(const IJsonReader &reader, const char *fileName) override;
    void getJSON(rapidjson::Document &doc) const override;

#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    inline MoBenchmark &benchmark()           { return m_benchmark; }
#   endif

private:
    ConfigPrivate *d_ptr;
#   ifdef XMRIG_FEATURE_MO_BENCHMARK
    MoBenchmark m_benchmark;
#   endif
};


} /* namespace xmrig */


#endif /* XMRIG_CONFIG_H */
