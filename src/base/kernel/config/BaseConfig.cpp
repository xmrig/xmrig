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
 *
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
 */

#include "base/kernel/config/BaseConfig.h"
#include "3rdparty/fmt/core.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"
#include "base/io/log/Log.h"
#include "base/io/log/Tags.h"
#include "base/kernel/interfaces/IJsonReader.h"
#include "base/kernel/Process.h"
#include "base/kernel/Versions.h"
#include "base/net/dns/Dns.h"
#include "version.h"


#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#ifdef XMRIG_FEATURE_TLS
#   include <openssl/opensslv.h>
#endif

#ifdef XMRIG_FEATURE_HWLOC
#   include "backend/cpu/Cpu.h"
#endif


namespace xmrig {


const char *BaseConfig::kApi            = "api";
const char *BaseConfig::kApiId          = "id";
const char *BaseConfig::kApiWorkerId    = "worker-id";
const char *BaseConfig::kAutosave       = "autosave";
const char *BaseConfig::kBackground     = "background";
const char *BaseConfig::kColors         = "colors";
const char *BaseConfig::kDryRun         = "dry-run";
const char *BaseConfig::kHttp           = "http";
const char *BaseConfig::kLogFile        = "log-file";
const char *BaseConfig::kPrintTime      = "print-time";
const char *BaseConfig::kSyslog         = "syslog";
const char *BaseConfig::kTitle          = "title";
const char *BaseConfig::kUserAgent      = "user-agent";
const char *BaseConfig::kVerbose        = "verbose";
const char *BaseConfig::kWatch          = "watch";


#ifdef XMRIG_FEATURE_TLS
const char *BaseConfig::kTls            = "tls";
#endif


} // namespace xmrig


bool xmrig::BaseConfig::read(const IJsonReader &reader, const char *fileName)
{
    m_fileName = fileName;

    if (reader.isEmpty()) {
        return false;
    }

    m_autoSave          = reader.getBool(kAutosave, m_autoSave);
    m_background        = reader.getBool(kBackground, m_background);
    m_dryRun            = reader.getBool(kDryRun, m_dryRun);
    m_syslog            = reader.getBool(kSyslog, m_syslog);
    m_watch             = reader.getBool(kWatch, m_watch);
    m_logFile           = reader.getString(kLogFile);
    m_userAgent         = reader.getString(kUserAgent);
    m_printTime         = std::min(reader.getUint(kPrintTime, m_printTime), 3600U);
    m_title             = reader.getValue(kTitle);

#   ifdef XMRIG_FEATURE_TLS
    m_tls = reader.getValue(kTls);
#   endif

    Log::setColors(reader.getBool(kColors, Log::isColors()));
    setVerbose(reader.getValue(kVerbose));

    const auto &api = reader.getObject(kApi);
    if (api.IsObject()) {
        m_apiId       = Json::getString(api, kApiId);
        m_apiWorkerId = Json::getString(api, kApiWorkerId);
    }

    m_http.load(reader.getObject(kHttp));
    m_pools.load(reader);

    Dns::set(reader.getObject(DnsConfig::kField));

    return m_pools.active() > 0;
}


bool xmrig::BaseConfig::save()
{
    if (m_fileName.isNull()) {
        return false;
    }

    rapidjson::Document doc;
    getJSON(doc);

    if (Json::save(m_fileName, doc)) {
        LOG_NOTICE("%s " WHITE_BOLD("configuration saved to: \"%s\""), Tags::config(), m_fileName.data());
        return true;
    }

    return false;
}


void xmrig::BaseConfig::printVersions()
{
    const auto &versions = Process::versions();

    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13s") CYAN_BOLD("%s/%s") WHITE_BOLD(" %s/%s"), "ABOUT", APP_NAME, APP_VERSION, Versions::kCompiler, versions.get(Versions::kCompiler).data());

#   if defined(XMRIG_FEATURE_TLS)
    std::string libs = fmt::format("{}/{} ", Versions::kTls, versions.get(Versions::kTls));
#   else
    std::string libs;
#   endif

#   if defined(XMRIG_FEATURE_HWLOC)
    libs += Cpu::info()->backend();
#   endif

    Log::print(GREEN_BOLD(" * ") WHITE_BOLD("%-13slibuv/%s %s"), "LIBS", versions.get(Versions::kUv).data(), libs.c_str());
}


void xmrig::BaseConfig::setVerbose(const rapidjson::Value &value)
{
    if (value.IsBool()) {
        Log::setVerbose(value.GetBool() ? 1 : 0);
    }
    else if (value.IsUint()) {
        Log::setVerbose(value.GetUint());
    }
}
