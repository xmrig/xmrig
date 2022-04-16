/* XMRig
 * Copyright (c) 2018-2022 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2022 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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
#include "version.h"


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


const char *Versions::kApp          = "app";
const char *Versions::kBase         = "base";
const char *Versions::kFmt          = "fmt";
const char *Versions::kRapidjson    = "rapidjson";
const char *Versions::kUv           = "uv";

#if defined (__INTEL_LLVM_COMPILER)
    const char *Versions::kCompiler = "icx";
#elif defined(__INTEL_COMPILER)
    const char *Versions::kCompiler = "icc";
#elif defined(__clang__)
    const char *Versions::kCompiler = "clang";
#elif defined(__GNUC__)
    const char *Versions::kCompiler = "gcc";
#elif defined(_MSC_VER)
#   if (_MSC_VER >= 1930)
#       define MSVC_VERSION 2022
#   elif (_MSC_VER >= 1920 && _MSC_VER < 1930)
#       define MSVC_VERSION 2019
#   elif (_MSC_VER >= 1910 && _MSC_VER < 1920)
#       define MSVC_VERSION 2017
#   elif _MSC_VER == 1900
#       define MSVC_VERSION 2015
#   else
#       define MSVC_VERSION 0
#   endif
    const char *Versions::kCompiler = "MSVC";
#else
    const char *Versions::kCompiler = "unknown";
#endif


#ifdef XMRIG_FEATURE_HTTP
const char *Versions::kLlhttp       = "llhttp";
#endif

#ifdef XMRIG_FEATURE_TLS
#   if defined(LIBRESSL_VERSION_TEXT)
    const char *Versions::kTls      = "libressl";
#   else
    const char *Versions::kTls      = "openssl";
#   endif
#endif

#ifdef XMRIG_FEATURE_SODIUM
const char *Versions::kSodium       = "sodium";
#endif

#ifdef XMRIG_FEATURE_SQLITE
const char *Versions::kSqlite       = "sqlite";
#endif

#ifdef XMRIG_FEATURE_HWLOC
const char *Versions::kHwloc        = "hwloc";
#endif

#ifdef XMRIG_FEATURE_POSTGRESQL
const char *kPq                     = "pq";
#endif


} // namespace xmrig


xmrig::Versions::Versions()
{
    m_data.insert({ kApp,           APP_VERSION });
    m_data.insert({ kBase,          fmt::format("{}.{}.{}", XMRIG_BASE_VERSION / 10000, XMRIG_BASE_VERSION / 100 % 100, XMRIG_BASE_VERSION % 100).c_str() });
    m_data.insert({ kUv,            uv_version_string() });
    m_data.insert({ kRapidjson,     RAPIDJSON_VERSION_STRING });
    m_data.insert({ kFmt,           fmt::format("{}.{}.{}", FMT_VERSION / 10000, FMT_VERSION / 100 % 100, FMT_VERSION % 100).c_str() });

#   if defined (__INTEL_LLVM_COMPILER)
    m_data.insert({ kCompiler,      fmt::format("{}.{}.{}", __INTEL_LLVM_COMPILER / 10000, __INTEL_LLVM_COMPILER / 100 % 100, __INTEL_LLVM_COMPILER % 100).c_str() });
#   elif defined (__INTEL_COMPILER)
#       if (__INTEL_COMPILER >= 2020)
        m_data.insert({ kCompiler,  XMRIG_TOSTRING(__INTEL_COMPILER) });
#       else
        m_data.insert({ kCompiler,  fmt::format("{}.{}.{}", __INTEL_COMPILER / 100, __INTEL_COMPILER / 10 % 10, __INTEL_COMPILER % 10).c_str() });
#       endif
#   elif defined(__clang__)
    m_data.insert({ kCompiler,      XMRIG_TOSTRING(__clang_major__.__clang_minor__.__clang_patchlevel__) });
#   elif defined(__GNUC__)
    m_data.insert({ kCompiler,      XMRIG_TOSTRING(__GNUC__.__GNUC_MINOR__.__GNUC_PATCHLEVEL__) });
#   elif defined(_MSC_VER)
    m_data.insert({ kCompiler,      XMRIG_TOSTRING(MSVC_VERSION) });
#   endif

#   ifdef XMRIG_FEATURE_HTTP
    m_data.insert({ kLlhttp,        XMRIG_TOSTRING(LLHTTP_VERSION_MAJOR.LLHTTP_VERSION_MINOR.LLHTTP_VERSION_PATCH) });
#   endif

#   ifdef XMRIG_FEATURE_TLS
#   if defined(LIBRESSL_VERSION_TEXT)
    m_data.insert({ kTls,           String(LIBRESSL_VERSION_TEXT).split(' ')[1] });
#   elif defined(OPENSSL_VERSION_TEXT)
    m_data.insert({ kTls,           String(OPENSSL_VERSION_TEXT).split(' ')[1] });
#   endif
#   endif

#   ifdef XMRIG_FEATURE_SODIUM
    m_data.insert({ kSodium,        sodium_version_string() });
#   endif

#   ifdef XMRIG_FEATURE_SQLITE
    m_data.insert({ kSqlite,        sqlite3_libversion() });
#   endif

#   if defined(XMRIG_FEATURE_HWLOC) && defined(XMRIG_LEGACY)
    m_data.insert({ kHwloc,         String(Cpu::info()->backend()).split('/')[1] });
#   endif

#   ifdef XMRIG_FEATURE_POSTGRESQL
    m_data.insert({ kPq,            fmt::format("{}.{}", PQlibVersion() / 10000, PQlibVersion() % 100).c_str() });
#   endif
}


const xmrig::String &xmrig::Versions::get(const char *key) const
{
    static const String empty;

    const auto it = m_data.find(key);

    return it != m_data.end() ? it->second : empty;
}


rapidjson::Value xmrig::Versions::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    Value out(kObjectType);
    toJSON(out, doc);

    return out;
}


void xmrig::Versions::toJSON(rapidjson::Value &out, rapidjson::Document &doc) const
{
    auto &allocator  = doc.GetAllocator();
    const auto &data = get();

    for (const auto &kv : data) {
        out.AddMember(kv.first.toJSON(), kv.second.toJSON(), allocator);
    }
}
