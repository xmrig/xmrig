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

#include "base/kernel/Versions.h"
#include "3rdparty/fmt/core.h"
#include "3rdparty/rapidjson/document.h"
#include "base/kernel/version.h"


#include <uv.h>


#ifdef XMRIG_FEATURE_HTTP
#   include "3rdparty/llhttp/llhttp.h"
#endif

#ifdef XMRIG_FEATURE_TLS
#   include <openssl/opensslv.h>
#endif

#ifdef XMRIG_FEATURE_SODIUM
#   include <sodium.h>
#endif

#ifdef XMRIG_FEATURE_SQLITE
#   include "3rdparty/sqlite/sqlite3.h"
#endif

#ifdef XMRIG_FEATURE_HWLOC
#   include "backend/cpu/Cpu.h"
#endif

#ifdef XMRIG_FEATURE_POSTGRESQL
#   include <libpq-fe.h>
#endif


namespace xmrig {


std::map<String, String> Versions::m_data;


} // namespace xmrig


const std::map<xmrig::String, xmrig::String> &xmrig::Versions::get()
{
    if (m_data.empty()) {
        m_data.insert({ "base",         BASE_VERSION });
        m_data.insert({ "uv",           uv_version_string() });
        m_data.insert({ "rapidjson",    RAPIDJSON_VERSION_STRING });
        m_data.insert({ "fmt",          fmt::format("{}.{}.{}", FMT_VERSION / 10000, FMT_VERSION / 100 % 100, FMT_VERSION % 100).c_str() });

#       ifdef XMRIG_FEATURE_HTTP
        m_data.insert({ "llhttp", XMRIG_TOSTRING(LLHTTP_VERSION_MAJOR.LLHTTP_VERSION_MINOR.LLHTTP_VERSION_PATCH) });
#       endif

#       ifdef XMRIG_FEATURE_TLS
#       if defined(LIBRESSL_VERSION_TEXT)
        m_data.insert({ "libressl", String(LIBRESSL_VERSION_TEXT).split(' ')[1] });
#       elif defined(OPENSSL_VERSION_TEXT)
        m_data.insert({ "openssl", String(OPENSSL_VERSION_TEXT).split(' ')[1] });
#       endif
#       endif

#       ifdef XMRIG_FEATURE_SODIUM
        m_data.insert({ "sodium", sodium_version_string() });
#       endif

#       ifdef XMRIG_FEATURE_SQLITE
        m_data.insert({ "sqlite", sqlite3_libversion() });
#       endif

#       ifdef XMRIG_FEATURE_HWLOC
        m_data.insert({ "hwloc", String(Cpu::info()->backend()).split('/')[1] });
#       endif

#       ifdef XMRIG_FEATURE_POSTGRESQL
        m_data.insert({ "pq", fmt::format("{}.{}", PQlibVersion() / 10000, PQlibVersion() % 100).c_str() });
#       endif
    }

    return m_data;
}


rapidjson::Value xmrig::Versions::toJSON(rapidjson::Document &doc)
{
    using namespace rapidjson;

    Value out(kObjectType);
    toJSON(out, doc);

    return out;
}


void xmrig::Versions::toJSON(rapidjson::Value &out, rapidjson::Document &doc)
{
    auto &allocator  = doc.GetAllocator();
    const auto &data = get();

    for (const auto &kv : data) {
        out.AddMember(kv.first.toJSON(), kv.second.toJSON(), allocator);
    }
}
