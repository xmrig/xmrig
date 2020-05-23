/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
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


#include "base/net/tls/TlsConfig.h"
#include "3rdparty/rapidjson/document.h"
#include "base/io/json/Json.h"
#include "base/io/log/Log.h"
#include "base/net/tls/TlsGen.h"


namespace xmrig {


const char *TlsConfig::kCert            = "cert";
const char *TlsConfig::kEnabled         = "enabled";
const char *TlsConfig::kCertKey         = "cert_key";
const char *TlsConfig::kCiphers         = "ciphers";
const char *TlsConfig::kCipherSuites    = "ciphersuites";
const char *TlsConfig::kDhparam         = "dhparam";
const char *TlsConfig::kGen             = "gen";
const char *TlsConfig::kProtocols       = "protocols";

static const char *kTLSv1               = "TLSv1";
static const char *kTLSv1_1             = "TLSv1.1";
static const char *kTLSv1_2             = "TLSv1.2";
static const char *kTLSv1_3             = "TLSv1.3";


} // namespace xmrig


/**
 * "cert"         load TLS certificate chain from file.
 * "cert_key"     load TLS private key from file.
 * "ciphers"      set list of available ciphers (TLSv1.2 and below).
 * "ciphersuites" set list of available TLSv1.3 ciphersuites.
 * "dhparam"      load DH parameters for DHE ciphers from file.
 */
xmrig::TlsConfig::TlsConfig(const rapidjson::Value &value)
{
    if (value.IsObject()) {
        m_enabled = Json::getBool(value, kEnabled, m_enabled);

        setProtocols(Json::getString(value, kProtocols));
        setCert(Json::getString(value, kCert));
        setKey(Json::getString(value, kCertKey));
        setCiphers(Json::getString(value, kCiphers));
        setCipherSuites(Json::getString(value, kCipherSuites));
        setDH(Json::getString(value, kDhparam));

        if (m_key.isNull()) {
            setKey(Json::getString(value, "cert-key"));
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
#   ifdef XMRIG_FORCE_TLS
    else if (value.IsNull()) {
        generate();
    }
#   endif
    else if (value.IsString()) {
        generate(value.GetString());
    }
    else {
        m_enabled = false;
    }
}


bool xmrig::TlsConfig::generate(const char *commonName)
{
    TlsGen gen;

    try {
        gen.generate(commonName);
    }
    catch (std::exception &ex) {
        LOG_ERR("%s", ex.what());

        return false;
    }

    setCert(gen.cert());
    setKey(gen.certKey());

    m_enabled = true;

    return true;
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
    obj.AddMember(StringRef(kCipherSuites), m_cipherSuites.toJSON(), allocator);
    obj.AddMember(StringRef(kDhparam),      m_dhparam.toJSON(), allocator);

    return obj;
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

    if (protocols.IsUint()) {
        return setProtocols(protocols.GetUint());
    }

    if (protocols.IsString()) {
        return setProtocols(protocols.GetString());
    }
}
