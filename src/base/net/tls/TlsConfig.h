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

#ifndef XMRIG_TLSCONFIG_H
#define XMRIG_TLSCONFIG_H


#include "base/tools/String.h"


namespace xmrig {


class Arguments;


class TlsConfig
{
public:
    static const char *kCert;
    static const char *kCertKey;
    static const char *kCiphers;
    static const char *kCipherSuites;
    static const char *kDhparam;
    static const char *kEnabled;
    static const char *kField;
    static const char *kGen;
    static const char *kProtocols;

    enum Versions {
        TLSv1   = 1,
        TLSv1_1 = 2,
        TLSv1_2 = 4,
        TLSv1_3 = 8
    };

    TlsConfig() = default;
    TlsConfig(const Arguments &arguments);

    inline TlsConfig(const rapidjson::Value &value, const TlsConfig &current)   { init(value, current); }
    inline TlsConfig(const rapidjson::Value &value)                             { init(value, {}); }

    inline bool isEnabled() const                           { return m_enabled; }
    inline bool isValid() const                             { return !m_cert.isEmpty() && !m_key.isEmpty(); }
    inline const String &cert() const                       { return m_cert; }
    inline const String &ciphers() const                    { return m_ciphers; }
    inline const String &ciphersuites() const               { return m_ciphersuites; }
    inline const String &dhparam() const                    { return m_dhparam; }
    inline const String &key() const                        { return m_key; }
    inline uint32_t protocols() const                       { return m_protocols; }

    inline bool operator!=(const TlsConfig &other) const    { return !isEqual(other); }
    inline bool operator==(const TlsConfig &other) const    { return isEqual(other); }

    bool generate(const char *commonName = nullptr);
    bool isEqual(const TlsConfig &other) const;
    rapidjson::Value toJSON(rapidjson::Document &doc) const;
    void print() const;

private:
    void init(const rapidjson::Value &value, const TlsConfig &current);
    void setProtocols(const char *protocols);
    void setProtocols(const rapidjson::Value &protocols);

    bool m_enabled       = false;
    uint32_t m_protocols = 0;
    String m_cert;
    String m_ciphers;
    String m_ciphersuites;
    String m_dhparam;
    String m_key;
};


} // namespace xmrig


#endif // XMRIG_TLSCONFIG_H
