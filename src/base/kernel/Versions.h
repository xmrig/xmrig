/* XMRig
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_VERSIONS_H
#define XMRIG_VERSIONS_H


#include "base/tools/String.h"


#include <map>


namespace xmrig {


class Versions
{
public:
    static const char *kApp;
    static const char *kBase;
    static const char *kCompiler;
    static const char *kFmt;
    static const char *kRapidjson;
    static const char *kUv;

#   ifdef XMRIG_FEATURE_HTTP
    static const char *kLlhttp;
#   endif

#   ifdef XMRIG_FEATURE_TLS
    static const char *kTls;
#   endif

#   ifdef XMRIG_FEATURE_SODIUM
    static const char *kSodium;
#   endif

#   ifdef XMRIG_FEATURE_SQLITE
    static const char *kSqlite;
#   endif

#   ifdef XMRIG_FEATURE_HWLOC
    static const char *kHwloc;
#   endif

#   ifdef XMRIG_FEATURE_POSTGRESQL
    static const char *kPq;
#   endif

    Versions();

    inline const std::map<String, String> &get() const          { return m_data; }
    inline const String &operator[](const char *key) const      { return get(key); }

    const String &get(const char *key) const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    void toJSON(rapidjson::Value &out, rapidjson::Document &doc) const;

private:
    std::map<String, String> m_data;
};


} /* namespace xmrig */


#endif /* XMRIG_VERSIONS_H */
