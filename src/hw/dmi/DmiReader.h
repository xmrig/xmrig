/* XMRig
 * Copyright (c) 2000-2002 Alan Cox     <alan@redhat.com>
 * Copyright (c) 2005-2020 Jean Delvare <jdelvare@suse.de>
 * Copyright (c) 2018-2021 SChernykh    <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig        <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_DMIREADER_H
#define XMRIG_DMIREADER_H


#include "hw/dmi/DmiBoard.h"
#include "hw/dmi/DmiMemory.h"


#include <functional>


namespace xmrig {


class DmiReader
{
public:
    DmiReader() = default;

    inline const DmiBoard &board() const                { return m_board; }
    inline const DmiBoard &system() const               { return m_system; }
    inline const std::vector<DmiMemory> &memory() const { return m_memory; }
    inline uint32_t size() const                        { return m_size; }
    inline uint32_t version() const                     { return m_version; }

    bool read();

#   ifdef XMRIG_FEATURE_API
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    void toJSON(rapidjson::Value &out, rapidjson::Document &doc) const;
#   endif

private:
    using Cleanup = std::function<void()>;

    bool decode(uint8_t *buf, const Cleanup &cleanup);
    bool decode(uint8_t *buf);

    DmiBoard m_board;
    DmiBoard m_system;
    std::vector<DmiMemory> m_memory;
    uint32_t m_size     = 0;
    uint32_t m_version  = 0;
};


} /* namespace xmrig */


#endif /* XMRIG_DMIREADER_H */
