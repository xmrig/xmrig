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


#include "crypto/rx/RxConfig.h"
#include "backend/cpu/Cpu.h"
#include "rapidjson/document.h"


#include <array>
#include <algorithm>


#ifdef _MSC_VER
#   define strcasecmp  _stricmp
#endif


namespace xmrig {


static const std::array<const char *, RxConfig::ModeMax> modeNames = { "auto", "fast", "light" };


} // namespace xmrig


const char *xmrig::RxConfig::modeName() const
{
    return modeNames[m_mode];
}


uint32_t xmrig::RxConfig::threads() const
{
    return m_threads < 1 ? static_cast<uint32_t>(Cpu::info()->threads()) : static_cast<uint32_t>(m_threads);
}


xmrig::RxConfig::Mode xmrig::RxConfig::readMode(const rapidjson::Value &value) const
{
    if (value.IsUint()) {
        return static_cast<Mode>(std::min(value.GetUint(), ModeMax - 1));
    }

    if (value.IsString()) {
        auto mode = value.GetString();

        for (size_t i = 0; i < modeNames.size(); i++) {
            if (strcasecmp(mode, modeNames[i]) == 0) {
                return static_cast<Mode>(i);
            }
        }
    }

    return AutoMode;
}
