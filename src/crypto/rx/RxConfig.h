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

#ifndef XMRIG_RXCONFIG_H
#define XMRIG_RXCONFIG_H


#include "3rdparty/rapidjson/fwd.h"


#ifdef XMRIG_FEATURE_MSR
#   include "hw/msr/MsrItem.h"
#endif


#include <vector>


namespace xmrig {


class RxConfig
{
public:
    enum Mode : uint32_t {
        AutoMode,
        FastMode,
        LightMode,
        ModeMax
    };

    enum ScratchpadPrefetchMode : uint32_t {
        ScratchpadPrefetchOff,
        ScratchpadPrefetchT0,
        ScratchpadPrefetchNTA,
        ScratchpadPrefetchMov,
        ScratchpadPrefetchMax,
    };

    static const char *kCacheQoS;
    static const char *kField;
    static const char *kInit;
    static const char *kInitAVX2;
    static const char *kMode;
    static const char *kOneGbPages;
    static const char *kRdmsr;
    static const char *kScratchpadPrefetchMode;
    static const char *kWrmsr;

#   ifdef XMRIG_FEATURE_HWLOC
    static const char *kNUMA;
#   endif

    bool read(const rapidjson::Value &value);
    rapidjson::Value toJSON(rapidjson::Document &doc) const;

#   ifdef XMRIG_FEATURE_HWLOC
    std::vector<uint32_t> nodeset() const;
#   else
    inline std::vector<uint32_t> nodeset() const { return std::vector<uint32_t>(); }
#   endif

    const char *modeName() const;
    uint32_t threads(uint32_t limit = 100) const;

    inline int initDatasetAVX2() const  { return m_initDatasetAVX2; }
    inline bool isOneGbPages() const    { return m_oneGbPages; }
    inline bool rdmsr() const           { return m_rdmsr; }
    inline bool wrmsr() const           { return m_wrmsr; }
    inline bool cacheQoS() const        { return m_cacheQoS; }
    inline Mode mode() const            { return m_mode; }

    inline ScratchpadPrefetchMode scratchpadPrefetchMode() const { return m_scratchpadPrefetchMode; }

#   ifdef XMRIG_FEATURE_MSR
    const char *msrPresetName() const;
    const MsrItems &msrPreset() const;
#   endif

private:
#   ifdef XMRIG_FEATURE_MSR
    uint32_t msrMod() const;
    void readMSR(const rapidjson::Value &value);

    bool m_wrmsr = true;
    MsrItems m_msrPreset;
#   else
    bool m_wrmsr = false;
#   endif

    bool m_cacheQoS = false;

    Mode readMode(const rapidjson::Value &value) const;

    bool m_numa           = true;
    bool m_oneGbPages     = false;
    bool m_rdmsr          = true;
    int m_threads         = -1;
    int m_initDatasetAVX2 = -1;
    Mode m_mode           = AutoMode;

    ScratchpadPrefetchMode m_scratchpadPrefetchMode = ScratchpadPrefetchT0;

#   ifdef XMRIG_FEATURE_HWLOC
    std::vector<uint32_t> m_nodeset;
#   endif

};


} /* namespace xmrig */


#endif /* XMRIG_RXCONFIG_H */
