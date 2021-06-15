/* XMRig
 * Copyright 2012-2013 The Cryptonote developers
 * Copyright 2014-2021 The Monero Project
 * Copyright 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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


#include "base/crypto/keccak.h"
#include "base/tools/cryptonote/Signatures.h"
#include "base/tools/cryptonote/crypto-ops.h"
#include "base/tools/Cvt.h"


struct ec_scalar { char data[32]; };
struct hash { char data[32]; };
struct ec_point { char data[32]; };
struct signature { ec_scalar c, r; };
struct s_comm { hash h; ec_point key; ec_point comm; };


static bool less32(const uint8_t* k0, const uint8_t* k1)
{
    for (int n = 31; n >= 0; --n)
    {
        if (k0[n] < k1[n])
            return true;
        if (k0[n] > k1[n])
            return false;
    }
    return false;
}


static void random32_unbiased(uint8_t* bytes)
{
    // l = 2^252 + 27742317777372353535851937790883648493.
    // l fits 15 times in 32 bytes (iow, 15 l is the highest multiple of l that fits in 32 bytes)
    static const uint8_t limit[32] = { 0xe3, 0x6a, 0x67, 0x72, 0x8b, 0xce, 0x13, 0x29, 0x8f, 0x30, 0x82, 0x8c, 0x0b, 0xa4, 0x10, 0x39, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0 };

    do {
        xmrig::Cvt::randomBytes(bytes, 32);
        if (!less32(bytes, limit)) {
            continue;
        }
        sc_reduce32(bytes);
    } while (!sc_isnonzero(bytes));
}


/* generate a random 32-byte (256-bit) integer and copy it to res */
static void random_scalar(ec_scalar& res)
{
    random32_unbiased((uint8_t*) res.data);
}


static void hash_to_scalar(const void* data, size_t length, ec_scalar& res)
{
    uint8_t md[200];
    xmrig::keccak((const char*) data, length, md);
    memcpy(&res, md, sizeof(res));
    sc_reduce32((uint8_t*) &res);
}


static void derivation_to_scalar(const uint8_t* derivation, size_t output_index, ec_scalar& res)
{
    struct {
        uint8_t derivation[32];
        uint8_t output_index[(sizeof(size_t) * 8 + 6) / 7];
    } buf;

    uint8_t* end = buf.output_index;
    memcpy(buf.derivation, derivation, sizeof(buf.derivation));

    size_t k = output_index;
    while (k >= 0x80) {
        *(end++) = (static_cast<uint8_t>(k) & 0x7F) | 0x80;
        k >>= 7;
    }
    *(end++) = static_cast<uint8_t>(k);

    hash_to_scalar(&buf, end - reinterpret_cast<uint8_t*>(&buf), res);
}


namespace xmrig {


void generate_signature(const uint8_t* prefix_hash, const uint8_t* pub, const uint8_t* sec, uint8_t* sig_bytes)
{
    ge_p3 tmp3;
    ec_scalar k;
    s_comm buf;

    memcpy(buf.h.data, prefix_hash, sizeof(buf.h.data));
    memcpy(buf.key.data, pub, sizeof(buf.key.data));

    signature& sig = *reinterpret_cast<signature*>(sig_bytes);

    do {
        random_scalar(k);
        ge_scalarmult_base(&tmp3, (unsigned char*)&k);
        ge_p3_tobytes((unsigned char*)&buf.comm, &tmp3);
        hash_to_scalar(&buf, sizeof(s_comm), sig.c);

        if (!sc_isnonzero((const unsigned char*)sig.c.data)) {
            continue;
        }

        sc_mulsub((unsigned char*)&sig.r, (unsigned char*)&sig.c, sec, (unsigned char*)&k);
    } while (!sc_isnonzero((const unsigned char*)sig.r.data));
}


bool check_signature(const uint8_t* prefix_hash, const uint8_t* pub, const uint8_t* sig_bytes)
{
    ge_p2 tmp2;
    ge_p3 tmp3;
    ec_scalar c;
    s_comm buf;

    memcpy(buf.h.data, prefix_hash, sizeof(buf.h.data));
    memcpy(buf.key.data, pub, sizeof(buf.key.data));

    if (ge_frombytes_vartime(&tmp3, pub) != 0) {
        return false;
    }

    const signature& sig = *reinterpret_cast<const signature*>(sig_bytes);

    if (sc_check((const uint8_t*)&sig.c) != 0 || sc_check((const uint8_t*)&sig.r) != 0 || !sc_isnonzero((const uint8_t*)&sig.c)) {
        return false;
    }

    ge_double_scalarmult_base_vartime(&tmp2, (const uint8_t*)&sig.c, &tmp3, (const uint8_t*)&sig.r);
    ge_tobytes((uint8_t*)&buf.comm, &tmp2);

    static const ec_point infinity = { { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} };
    if (memcmp(&buf.comm, &infinity, 32) == 0) {
        return false;
    }

    hash_to_scalar(&buf, sizeof(s_comm), c);
    sc_sub((uint8_t*)&c, (uint8_t*)&c, (const uint8_t*)&sig.c);

    return sc_isnonzero((uint8_t*)&c) == 0;
}


bool generate_key_derivation(const uint8_t* key1, const uint8_t* key2, uint8_t* derivation)
{
    ge_p3 point;
    ge_p2 point2;
    ge_p1p1 point3;

    if (ge_frombytes_vartime(&point, key1) != 0) {
        return false;
    }

    ge_scalarmult(&point2, key2, &point);
    ge_mul8(&point3, &point2);
    ge_p1p1_to_p2(&point2, &point3);
    ge_tobytes(derivation, &point2);

    return true;
}


void derive_secret_key(const uint8_t* derivation, size_t output_index, const uint8_t* base, uint8_t* derived_key)
{
    ec_scalar scalar;

    derivation_to_scalar(derivation, output_index, scalar);
    sc_add(derived_key, base, (uint8_t*) &scalar);
}


} /* namespace xmrig */
