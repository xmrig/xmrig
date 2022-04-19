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

#include "base/net/tls/TlsContext.h"
#include "3rdparty/fmt/core.h"
#include "base/io/Env.h"
#include "base/net/tls/TlsConfig.h"


#include <openssl/err.h>
#include <openssl/ssl.h>
#include <stdexcept>


// https://wiki.openssl.org/index.php/OpenSSL_1.1.0_Changes#Compatibility_Layer
#if OPENSSL_VERSION_NUMBER < 0x10100000L
int DH_set0_pqg(DH *dh, BIGNUM *p, BIGNUM *q, BIGNUM *g)
{
    assert(q == nullptr);

    dh->p = p;
    dh->g = g;

    return 1;
 }
#endif


namespace xmrig {


class TlsContext::Private
{
public:
    void load(const TlsConfig &config);

    SSL_CTX *ctx    = nullptr;

private:
    static inline const char *error()   { return ERR_reason_error_string(ERR_get_error()); }

    static DH *get_dh2048();

    void setDH(const String &dhparam) const;
    void setProtocols(uint32_t protocols) const;
};


} // namespace xmrig


xmrig::TlsContext::~TlsContext()
{
    SSL_CTX_free(d->ctx);
}


SSL_CTX *xmrig::TlsContext::handle() const
{
    return d->ctx;
}


xmrig::TlsContext::TlsContext() :
    d(std::make_shared<Private>())
{
}


std::shared_ptr<xmrig::TlsContext> xmrig::TlsContext::create(const TlsConfig &config)
{
    if (!config.isEnabled()) {
        return nullptr;
    }

    auto tls = std::shared_ptr<TlsContext>(new TlsContext());
    tls->d->load(config);

    return tls;
}


void xmrig::TlsContext::Private::load(const TlsConfig &config)
{
    if ((ctx = SSL_CTX_new(SSLv23_server_method())) == nullptr) {
        throw std::runtime_error("Unable to create SSL context");
    }

    const auto cert = Env::expand(config.cert());
    if (cert.isNull()) {
        throw std::runtime_error("Unable to load cert file");
    }

    if (SSL_CTX_use_certificate_chain_file(ctx, cert) <= 0) {
        throw std::runtime_error(fmt::format("Unable to load cert file \"{}\": \"{}\"", cert.data(), error()));
    }

    const auto key = Env::expand(config.key());
    if (key.isNull()) {
        throw std::runtime_error("Unable to load key file");
    }

    if (SSL_CTX_use_PrivateKey_file(ctx, key, SSL_FILETYPE_PEM) <= 0) {
        throw std::runtime_error(fmt::format("Unable to load key file \"{}\": \"{}\"", key.data(), error()));
    }

    SSL_CTX_set_options(ctx, SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3);
    SSL_CTX_set_options(ctx, SSL_OP_CIPHER_SERVER_PREFERENCE);

#   if OPENSSL_VERSION_NUMBER >= 0x1010100fL && !defined(LIBRESSL_VERSION_NUMBER)
    SSL_CTX_set_max_early_data(ctx, 0);
#   endif

    setProtocols(config.protocols());

    if (!config.ciphers().isNull() && SSL_CTX_set_cipher_list(ctx, config.ciphers()) <= 0) {
        throw std::runtime_error(fmt::format("Unable to set cipher list: \"{}\"", error()));
    }

#   if OPENSSL_VERSION_NUMBER >= 0x1010100fL && !defined(LIBRESSL_VERSION_NUMBER)
    if (!config.ciphersuites().isNull() && SSL_CTX_set_ciphersuites(ctx, config.ciphersuites()) <= 0) {
        throw std::runtime_error(fmt::format("Unable to set ciphersuites: \"{}\"", error()));
    }
#   endif

    setDH(Env::expand(config.dhparam()));
}


// https://wiki.openssl.org/index.php/Diffie-Hellman_parameters
DH *xmrig::TlsContext::Private::get_dh2048()
{
    static unsigned char dhp_2048[] = {
        0xB2, 0x91, 0xA7, 0x05, 0x31, 0xCE, 0x12, 0x9D, 0x03, 0x43,
        0xAF, 0x13, 0xAF, 0x4B, 0x8E, 0x4C, 0x04, 0x13, 0x4F, 0x72,
        0x00, 0x73, 0x2C, 0x67, 0xC3, 0xE0, 0x50, 0xBF, 0x72, 0x5E,
        0xBE, 0x45, 0x89, 0x4C, 0x01, 0x45, 0xA6, 0x5E, 0xA7, 0xA8,
        0xDC, 0x2F, 0x1D, 0x91, 0x2D, 0x58, 0x0D, 0x71, 0x97, 0x3D,
        0xAE, 0xFE, 0x86, 0x29, 0x37, 0x5F, 0x5E, 0x6D, 0x81, 0x56,
        0x07, 0x83, 0xF2, 0xF8, 0xEC, 0x4E, 0xF8, 0x7A, 0xEC, 0xEA,
        0xD9, 0xEA, 0x61, 0x3C, 0xAF, 0x51, 0x30, 0xB7, 0xA7, 0x67,
        0x3F, 0x59, 0xAD, 0x2E, 0x23, 0x57, 0x64, 0xA2, 0x99, 0x15,
        0xBD, 0xD9, 0x8D, 0xBA, 0xE6, 0x8F, 0xFB, 0xB3, 0x77, 0x3B,
        0xE6, 0x5C, 0xC1, 0x03, 0xCF, 0x38, 0xD4, 0xF6, 0x2E, 0x0B,
        0xF3, 0x20, 0xBE, 0xF0, 0xFC, 0x85, 0xEF, 0x5F, 0xCE, 0x0E,
        0x42, 0x17, 0x3B, 0x72, 0x43, 0x4C, 0x3A, 0xF5, 0xC8, 0xB4,
        0x40, 0x52, 0x03, 0x72, 0x9A, 0x2C, 0xA4, 0x23, 0x2A, 0xA2,
        0x52, 0xA3, 0xC2, 0x76, 0x08, 0x1C, 0x2E, 0x60, 0x44, 0xE4,
        0x12, 0x5D, 0x80, 0x47, 0x6C, 0x7A, 0x5A, 0x8E, 0x18, 0xC9,
        0x8C, 0x22, 0xC8, 0x07, 0x75, 0xE2, 0x77, 0x3A, 0x90, 0x2E,
        0x79, 0xC3, 0xF5, 0x4E, 0x4E, 0xDE, 0x14, 0x29, 0xA4, 0x5B,
        0x32, 0xCC, 0xE5, 0x05, 0x09, 0x2A, 0xC9, 0x1C, 0xB4, 0x8E,
        0x99, 0xCF, 0x57, 0xF2, 0x1B, 0x5F, 0x18, 0x89, 0x29, 0xF2,
        0xB0, 0xF3, 0xAC, 0x67, 0x16, 0x90, 0x4A, 0x1D, 0xD6, 0xF5,
        0x84, 0x71, 0x1D, 0x0E, 0x61, 0x5F, 0xE2, 0x2D, 0x52, 0x87,
        0x0D, 0x8F, 0x84, 0xCB, 0xFC, 0xF0, 0x5D, 0x4C, 0x9F, 0x59,
        0xA9, 0xD6, 0x83, 0x70, 0x4B, 0x98, 0x6A, 0xCA, 0x78, 0x53,
        0x27, 0x32, 0x59, 0x35, 0x0A, 0xB8, 0x29, 0x18, 0xAF, 0x58,
        0x45, 0x63, 0xEB, 0x43, 0x28, 0x7B
    };

    static unsigned char dhg_2048[] = { 0x02 };


    auto dh = DH_new();
    if (dh == nullptr) {
        return nullptr;
    }

    auto p = BN_bin2bn(dhp_2048, sizeof(dhp_2048), nullptr);
    auto g = BN_bin2bn(dhg_2048, sizeof(dhg_2048), nullptr);

    if (p == nullptr || g == nullptr || !DH_set0_pqg(dh, p, nullptr, g)) {
        DH_free(dh);
        BN_free(p);
        BN_free(g);

        return nullptr;
    }

    return dh;
}


void xmrig::TlsContext::Private::setDH(const String &dhparam) const
{
    DH *dh = nullptr;

    if (!dhparam.isEmpty()) {
        BIO *bio = BIO_new_file(dhparam, "r");
        if (bio) {
            dh = PEM_read_bio_DHparams(bio, nullptr, nullptr, nullptr);
            BIO_free(bio);
        }

        if (!dh) {
            throw std::runtime_error(fmt::format("Unable to load DH params \"{}\": \"{}\"", dhparam.data(), error()));
        }
    }
    else {
        dh = get_dh2048();
    }

    const int rc = SSL_CTX_set_tmp_dh(ctx, dh); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    DH_free(dh);

    if (rc <= 0) {
        throw std::runtime_error(fmt::format("Unable to set DH params: \"{}\"", error()));
    }
}


void xmrig::TlsContext::Private::setProtocols(uint32_t protocols) const
{
    if (protocols == 0) {
        return;
    }

    if (!(protocols & TlsConfig::TLSv1)) {
        SSL_CTX_set_options(ctx, SSL_OP_NO_TLSv1);
    }

#   ifdef SSL_OP_NO_TLSv1_1
    SSL_CTX_clear_options(ctx, SSL_OP_NO_TLSv1_1);
    if (!(protocols & TlsConfig::TLSv1_1)) {
        SSL_CTX_set_options(ctx, SSL_OP_NO_TLSv1_1);
    }
#   endif

#   ifdef SSL_OP_NO_TLSv1_2
    SSL_CTX_clear_options(ctx, SSL_OP_NO_TLSv1_2);
    if (!(protocols & TlsConfig::TLSv1_2)) {
        SSL_CTX_set_options(ctx, SSL_OP_NO_TLSv1_2);
    }
#   endif

#   ifdef SSL_OP_NO_TLSv1_3
    SSL_CTX_clear_options(ctx, SSL_OP_NO_TLSv1_3);
    if (!(protocols & TlsConfig::TLSv1_3)) {
        SSL_CTX_set_options(ctx, SSL_OP_NO_TLSv1_3);
    }
#   endif
}
