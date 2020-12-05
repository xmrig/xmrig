/* XMRig
 * Copyright (c) 2018-2020 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2020 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_CVT_H
#define XMRIG_CVT_H


#include "3rdparty/rapidjson/fwd.h"
#include "base/tools/Buffer.h"
#include "base/tools/String.h"


#include <string>


namespace xmrig {


class Cvt
{
public:
    inline static bool fromHex(Buffer &buf, const String &hex)                  { return fromHex(buf, hex.data(), hex.size()); }
    inline static Buffer fromHex(const std::string &hex)                        { return fromHex(hex.data(), hex.size()); }
    inline static Buffer fromHex(const String &hex)                             { return fromHex(hex.data(), hex.size()); }
    inline static String toHex(const Buffer &data)                              { return toHex(data.data(), data.size()); }
    inline static String toHex(const std::string &data)                         { return toHex(reinterpret_cast<const uint8_t *>(data.data()), data.size()); }

    static bool fromHex(Buffer &buf, const char *in, size_t size);
    static bool fromHex(Buffer &buf, const rapidjson::Value &value);
    static bool fromHex(std::string &buf, const char *in, size_t size);
    static bool fromHex(uint8_t *bin, size_t bin_maxlen, const char *hex, size_t hex_len);
    static bool fromHex(uint8_t *bin, size_t bin_maxlen, const rapidjson::Value &value);
    static bool toHex(char *hex, size_t hex_maxlen, const uint8_t *bin, size_t bin_len);
    static Buffer fromHex(const char *in, size_t size);
    static Buffer randomBytes(size_t size);
    static rapidjson::Value toHex(const Buffer &data, rapidjson::Document &doc);
    static rapidjson::Value toHex(const std::string &data, rapidjson::Document &doc);
    static rapidjson::Value toHex(const uint8_t *in, size_t size, rapidjson::Document &doc);
    static String toHex(const uint8_t *in, size_t size);
};


} /* namespace xmrig */


#endif /* XMRIG_CVT_H */
