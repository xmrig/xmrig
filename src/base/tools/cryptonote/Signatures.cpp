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

extern "C" {

#include "base/tools/cryptonote/crypto-ops.h"

}

#include "base/tools/Cvt.h"

#ifdef XMRIG_PROXY_PROJECT
#define PROFILE_SCOPE(x)
#else
#include "crypto/rx/Profiler.h"
#endif


struct ec_scalar { char data[32]; };
struct hash { char data[32]; };
struct ec_point { char data[32]; };
struct signature { ec_scalar c, r; };
struct s_comm { hash h; ec_point key; ec_point comm; };


static inline void random_scalar(ec_scalar& res)
{
    // Don't care about bias or possible 0 after reduce: probability ~10^-76, not happening in this universe.
    // Performance matters more. It's a miner after all.
    xmrig::Cvt::randomBytes(res.data, sizeof(res.data));
    sc_reduce32((uint8_t*) res.data);
}


static void hash_to_scalar(const void* data, size_t length, ec_scalar& res)
{
    xmrig::keccak((const uint8_t*) data, length, (uint8_t*) &res, sizeof(res));
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
    PROFILE_SCOPE(GenerateSignature);

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


bool generate_key_derivation(const uint8_t* key1, const uint8_t* key2, uint8_t* derivation, uint8_t* view_tag)
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

    if (view_tag) {
        constexpr uint8_t salt[] = "view_tag";
        constexpr size_t SALT_SIZE = sizeof(salt) - 1;

        uint8_t buf[SALT_SIZE + 32 + 1];
        memcpy(buf, salt, SALT_SIZE);
        memcpy(buf + SALT_SIZE, derivation, 32);

        // Assuming output_index == 0
        buf[SALT_SIZE + 32] = 0;

        uint8_t view_tag_full[32];
        xmrig::keccak(buf, sizeof(buf), view_tag_full, sizeof(view_tag_full));
        *view_tag = view_tag_full[0];
    }

    return true;
}


void derive_secret_key(const uint8_t* derivation, size_t output_index, const uint8_t* base, uint8_t* derived_key)
{
    ec_scalar scalar;

    derivation_to_scalar(derivation, output_index, scalar);
    sc_add(derived_key, base, (uint8_t*) &scalar);
}


bool derive_public_key(const uint8_t* derivation, size_t output_index, const uint8_t* base, uint8_t* derived_key)
{
    ec_scalar scalar;
    ge_p3 point1;
    ge_p3 point2;
    ge_cached point3;
    ge_p1p1 point4;
    ge_p2 point5;

    if (ge_frombytes_vartime(&point1, base) != 0) {
        return false;
    }

    derivation_to_scalar(derivation, output_index, scalar);
    ge_scalarmult_base(&point2, (uint8_t*) &scalar);
    ge_p3_to_cached(&point3, &point2);
    ge_add(&point4, &point1, &point3);
    ge_p1p1_to_p2(&point5, &point4);
    ge_tobytes(derived_key, &point5);

    return true;
}


void derive_view_secret_key(const uint8_t* spend_secret_key, uint8_t* view_secret_key)
{
    keccak(spend_secret_key, 32, view_secret_key, 32);
    sc_reduce32(view_secret_key);
}


void generate_keys(uint8_t* pub, uint8_t* sec)
{
    random_scalar(*((ec_scalar*)sec));

    ge_p3 point;
    ge_scalarmult_base(&point, sec);
    ge_p3_tobytes(pub, &point);
}


bool secret_key_to_public_key(const uint8_t* sec, uint8_t* pub)
{
    if (sc_check(sec) != 0) {
        return false;
    }

    ge_p3 point;
    ge_scalarmult_base(&point, sec);
    ge_p3_tobytes(pub, &point);

    return true;
}


} /* namespace xmrig */
