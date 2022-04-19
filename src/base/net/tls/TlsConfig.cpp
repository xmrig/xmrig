/* XMRig
 * Copyright (c) 2018      Lee Clagett <https://github.com/vtnerd>
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

#include "base/net/tls/TlsConfig.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"
#include "base/io/log/Log.h"
#include "base/net/tls/TlsGen.h"


#ifdef APP_DEBUG
#   include "base/io/log/Log.h"
#   include "base/kernel/Config.h"
#endif


namespace xmrig {


extern const char *tls_tag();


const char *TlsConfig::kCert            = "cert";
const char *TlsConfig::kEnabled         = "enabled";
const char *TlsConfig::kCertKey         = "cert-key";
const char *TlsConfig::kCiphers         = "ciphers";
const char *TlsConfig::kCipherSuites    = "ciphersuites";
const char *TlsConfig::kDhparam         = "dhparam";
const char *TlsConfig::kField           = "tls";
const char *TlsConfig::kGen             = "gen";
const char *TlsConfig::kProtocols       = "protocols";

static const char *kTLSv1               = "TLSv1";
static const char *kTLSv1_1             = "TLSv1.1";
static const char *kTLSv1_2             = "TLSv1.2";
static const char *kTLSv1_3             = "TLSv1.3";


} // namespace xmrig


xmrig::TlsConfig::TlsConfig(const Arguments &arguments)
{

}


//xmrig::TlsConfig::TlsConfig(const rapidjson::Value &value, const TlsConfig &current)
//{

//}


bool xmrig::TlsConfig::generate(const char *commonName)
{
    TlsGen gen;

    try {
        gen.generate(commonName);
    }
    catch (std::exception &ex) {
        m_enabled = false;

        LOG_ERR("%s " RED_BOLD("%s"), tls_tag(), ex.what());

        return false;
    }

    m_cert      = gen.cert();
    m_key       = gen.certKey();
    m_enabled   = true;

    return true;
}


bool xmrig::TlsConfig::isEqual(const TlsConfig &other) const
{
    return (m_enabled           == other.m_enabled
            && m_protocols      == other.m_protocols
            && m_cert           == other.m_cert
            && m_ciphers        == other.m_ciphers
            && m_ciphersuites   == other.m_ciphersuites
            && m_dhparam        == other.m_dhparam
            && m_key            == other.m_key
            );
}


rapidjson::Value xmrig::TlsConfig::toJSON(rapidjson::Document &doc) const
{
    using namespace rapidjson;

    auto &allocator = doc.GetAllocator();
    Value obj(kObjectType);
    obj.AddMember(StringRef(kEnabled), m_enabled, allocator);

    if (m_protocols > 0) {
        std::vector<String> protocols;

        if (m_protocols & TLSv1) {
            protocols.emplace_back(kTLSv1);
        }

        if (m_protocols & TLSv1_1) {
            protocols.emplace_back(kTLSv1_1);
        }

        if (m_protocols & TLSv1_2) {
            protocols.emplace_back(kTLSv1_2);
        }

        if (m_protocols & TLSv1_3) {
            protocols.emplace_back(kTLSv1_3);
        }

        obj.AddMember(StringRef(kProtocols), String::join(protocols, ' ').toJSON(doc), allocator);
    }
    else {
        obj.AddMember(StringRef(kProtocols), kNullType, allocator);
    }

    obj.AddMember(StringRef(kCert),         m_cert.toJSON(), allocator);
    obj.AddMember(StringRef(kCertKey),      m_key.toJSON(), allocator);
    obj.AddMember(StringRef(kCiphers),      m_ciphers.toJSON(), allocator);
    obj.AddMember(StringRef(kCipherSuites), m_ciphersuites.toJSON(), allocator);
    obj.AddMember(StringRef(kDhparam),      m_dhparam.toJSON(), allocator);

    return obj;
}


void xmrig::TlsConfig::print() const
{
#   ifdef APP_DEBUG
    LOG_DEBUG("%s " MAGENTA_BOLD("TLS")
              MAGENTA("<enabled=") CYAN("%d")
              MAGENTA(", protocols=") CYAN("%u")
              MAGENTA(", cert=") "\"%s\""
              MAGENTA(", ciphers=") "\"%s\""
              MAGENTA(", ciphersuites=") "\"%s\""
              MAGENTA(", dhparam=") "\"%s\""
              MAGENTA(", key=") "\"%s\""
              MAGENTA(">"),
              Config::tag(), m_enabled, m_protocols, m_cert.data(), m_ciphers.data(), m_ciphersuites.data(), m_dhparam.data(), m_key.data());
#   endif
}


void xmrig::TlsConfig::init(const rapidjson::Value &value, const TlsConfig &current)
{
    if (value.IsObject()) {
        setProtocols(Json::getString(value, kProtocols));

        m_enabled       = Json::getBool(value, kEnabled, current.m_enabled);
        m_cert          = Json::getString(value, kCert, current.m_cert);
        m_key           = Json::getString(value, kCertKey, current.m_key);
        m_ciphers       = Json::getString(value, kCiphers, current.m_ciphers);
        m_ciphersuites  = Json::getString(value, kCipherSuites, current.m_ciphersuites);
        m_dhparam       = Json::getString(value, kDhparam, current.m_dhparam);

        if (m_key.isNull()) {
            m_key = Json::getString(value, "cert_key", current.m_key);
        }

        if (m_enabled && !isValid()) {
            generate(Json::getString(value, kGen));
        }
    }
    else if (value.IsBool()) {
        m_enabled = value.GetBool();

        if (m_enabled) {
            generate();
        }
    }
    else if (value.IsString()) {
        generate(value.GetString());
    }
    else {
        m_enabled = false;
    }
}


void xmrig::TlsConfig::setProtocols(const char *protocols)
{
    const std::vector<String> vec = String(protocols).split(' ');

    for (const String &value : vec) {
        if (value == kTLSv1) {
            m_protocols |= TLSv1;
        }
        else if (value == kTLSv1_1) {
            m_protocols |= TLSv1_1;
        }
        else if (value == kTLSv1_2) {
            m_protocols |= TLSv1_2;
        }
        else if (value == kTLSv1_3) {
            m_protocols |= TLSv1_3;
        }
    }
}


void xmrig::TlsConfig::setProtocols(const rapidjson::Value &protocols)
{
    m_protocols = 0;

    if (protocols.IsString()) {
        setProtocols(protocols.GetString());
    } else if (protocols.IsUint()) {
        m_protocols = protocols.GetUint();
    }
}
