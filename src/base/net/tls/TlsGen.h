/* XMRig
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

#ifndef XMRIG_TLSGEN_H
#define XMRIG_TLSGEN_H


#include "base/tools/Object.h"
#include "base/tools/String.h"


using EVP_PKEY  = struct evp_pkey_st;
using X509      = struct x509_st;


namespace xmrig {


class TlsGen
{
public:
    XMRIG_DISABLE_COPY_MOVE(TlsGen)

    TlsGen() : m_cert("cert.pem"), m_certKey("cert_key.pem") {}
    ~TlsGen();

    inline const String &cert() const       { return m_cert; }
    inline const String &certKey() const    { return m_certKey; }

    void generate(const char *commonName = nullptr);

private:
    bool generate_x509(const char *commonName);
    bool write();

    const String m_cert;
    const String m_certKey;
    EVP_PKEY *m_pkey    = nullptr;
    X509 *m_x509        = nullptr;
};


} /* namespace xmrig */


#endif /* XMRIG_TLSGEN_H */
