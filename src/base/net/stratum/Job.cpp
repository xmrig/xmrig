/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2019      Howard Chu  <https://github.com/hyc>
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


#include <cassert>
#include <cstring>


#include "base/net/stratum/Job.h"
#include "base/tools/Buffer.h"
#include "base/tools/Cvt.h"
#include "base/tools/cryptonote/BlockTemplate.h"
#include "base/tools/cryptonote/Signatures.h"
#include "base/crypto/keccak.h"


xmrig::Job::Job(bool nicehash, const Algorithm &algorithm, const String &clientId) :
    m_algorithm(algorithm),
    m_nicehash(nicehash),
    m_clientId(clientId)
{
}


bool xmrig::Job::isEqual(const Job &other) const
{
    return m_id == other.m_id && m_clientId == other.m_clientId && memcmp(m_blob, other.m_blob, sizeof(m_blob)) == 0;
}


bool xmrig::Job::setBlob(const char *blob)
{
    if (!blob) {
        return false;
    }

    m_size = strlen(blob);
    if (m_size % 2 != 0) {
        return false;
    }

    m_size /= 2;

    const size_t minSize = nonceOffset() + nonceSize();
    if (m_size < minSize || m_size >= sizeof(m_blob)) {
        return false;
    }

    if (!Cvt::fromHex(m_blob, sizeof(m_blob), blob, m_size * 2)) {
        return false;
    }

    if (*nonce() != 0 && !m_nicehash) {
        m_nicehash = true;
    }

#   ifdef XMRIG_PROXY_PROJECT
    memset(m_rawBlob, 0, sizeof(m_rawBlob));
    memcpy(m_rawBlob, blob, m_size * 2);
#   endif

    return true;
}


bool xmrig::Job::setSeedHash(const char *hash)
{
    if (!hash || (strlen(hash) != kMaxSeedSize * 2)) {
        return false;
    }

#   ifdef XMRIG_PROXY_PROJECT
    m_rawSeedHash = hash;
#   endif

    m_seed = Cvt::fromHex(hash, kMaxSeedSize * 2);

    return !m_seed.empty();
}


bool xmrig::Job::setTarget(const char *target)
{
    if (!target) {
        return false;
    }

    const auto raw    = Cvt::fromHex(target, strlen(target));
    const size_t size = raw.size();

    if (size == 4) {
        m_target = 0xFFFFFFFFFFFFFFFFULL / (0xFFFFFFFFULL / uint64_t(*reinterpret_cast<const uint32_t *>(raw.data())));
    }
    else if (size == 8) {
        m_target = *reinterpret_cast<const uint64_t *>(raw.data());
    }
    else {
        return false;
    }

#   ifdef XMRIG_PROXY_PROJECT
    assert(sizeof(m_rawTarget) > (size * 2));

    memset(m_rawTarget, 0, sizeof(m_rawTarget));
    memcpy(m_rawTarget, target, std::min(size * 2, sizeof(m_rawTarget)));
#   endif

    m_diff = toDiff(m_target);
    return true;
}


void xmrig::Job::setDiff(uint64_t diff)
{
    m_diff   = diff;
    m_target = toDiff(diff);

#   ifdef XMRIG_PROXY_PROJECT
    Cvt::toHex(m_rawTarget, sizeof(m_rawTarget), reinterpret_cast<uint8_t *>(&m_target), sizeof(m_target));
#   endif
}


void xmrig::Job::copy(const Job &other)
{
    m_algorithm  = other.m_algorithm;
    m_nicehash   = other.m_nicehash;
    m_size       = other.m_size;
    m_clientId   = other.m_clientId;
    m_id         = other.m_id;
    m_backend    = other.m_backend;
    m_diff       = other.m_diff;
    m_height     = other.m_height;
    m_target     = other.m_target;
    m_index      = other.m_index;
    m_seed       = other.m_seed;
    m_extraNonce = other.m_extraNonce;
    m_poolWallet = other.m_poolWallet;

    memcpy(m_blob, other.m_blob, sizeof(m_blob));

#   ifdef XMRIG_PROXY_PROJECT
    m_rawSeedHash = other.m_rawSeedHash;

    memcpy(m_rawBlob, other.m_rawBlob, sizeof(m_rawBlob));
    memcpy(m_rawTarget, other.m_rawTarget, sizeof(m_rawTarget));
#   endif

#   ifdef XMRIG_FEATURE_BENCHMARK
    m_benchSize = other.m_benchSize;
#   endif

#   ifdef XMRIG_PROXY_PROJECT
    memcpy(m_spendSecretKey, other.m_spendSecretKey, sizeof(m_spendSecretKey));
    memcpy(m_viewSecretKey, other.m_viewSecretKey, sizeof(m_viewSecretKey));
    memcpy(m_spendPublicKey, other.m_spendPublicKey, sizeof(m_spendPublicKey));
    memcpy(m_viewPublicKey, other.m_viewPublicKey, sizeof(m_viewPublicKey));
    m_minerTxPrefix = other.m_minerTxPrefix;
    m_minerTxEphPubKeyOffset = other.m_minerTxEphPubKeyOffset;
    m_minerTxPubKeyOffset = other.m_minerTxPubKeyOffset;
    m_minerTxMerkleTreeBranch = other.m_minerTxMerkleTreeBranch;
#   else
    memcpy(m_ephPublicKey, other.m_ephPublicKey, sizeof(m_ephPublicKey));
    memcpy(m_ephSecretKey, other.m_ephSecretKey, sizeof(m_ephSecretKey));
#   endif

    m_hasMinerSignature = other.m_hasMinerSignature;
}


void xmrig::Job::move(Job &&other)
{
    m_algorithm  = other.m_algorithm;
    m_nicehash   = other.m_nicehash;
    m_size       = other.m_size;
    m_clientId   = std::move(other.m_clientId);
    m_id         = std::move(other.m_id);
    m_backend    = other.m_backend;
    m_diff       = other.m_diff;
    m_height     = other.m_height;
    m_target     = other.m_target;
    m_index      = other.m_index;
    m_seed       = std::move(other.m_seed);
    m_extraNonce = std::move(other.m_extraNonce);
    m_poolWallet = std::move(other.m_poolWallet);

    memcpy(m_blob, other.m_blob, sizeof(m_blob));

    other.m_size        = 0;
    other.m_diff        = 0;
    other.m_algorithm   = Algorithm::INVALID;

#   ifdef XMRIG_PROXY_PROJECT
    m_rawSeedHash = std::move(other.m_rawSeedHash);

    memcpy(m_rawBlob, other.m_rawBlob, sizeof(m_rawBlob));
    memcpy(m_rawTarget, other.m_rawTarget, sizeof(m_rawTarget));
#   endif

#   ifdef XMRIG_FEATURE_BENCHMARK
    m_benchSize = other.m_benchSize;
#   endif

#   ifdef XMRIG_PROXY_PROJECT
    memcpy(m_spendSecretKey, other.m_spendSecretKey, sizeof(m_spendSecretKey));
    memcpy(m_viewSecretKey, other.m_viewSecretKey, sizeof(m_viewSecretKey));
    memcpy(m_spendPublicKey, other.m_spendPublicKey, sizeof(m_spendPublicKey));
    memcpy(m_viewPublicKey, other.m_viewPublicKey, sizeof(m_viewPublicKey));
    m_minerTxPrefix = std::move(other.m_minerTxPrefix);
    m_minerTxEphPubKeyOffset = other.m_minerTxEphPubKeyOffset;
    m_minerTxPubKeyOffset = other.m_minerTxPubKeyOffset;
    m_minerTxMerkleTreeBranch = std::move(other.m_minerTxMerkleTreeBranch);
#   else
    memcpy(m_ephPublicKey, other.m_ephPublicKey, sizeof(m_ephPublicKey));
    memcpy(m_ephSecretKey, other.m_ephSecretKey, sizeof(m_ephSecretKey));
#   endif

    m_hasMinerSignature = other.m_hasMinerSignature;
}


#ifdef XMRIG_PROXY_PROJECT


void xmrig::Job::setSpendSecretKey(uint8_t* key)
{
    m_hasMinerSignature = true;
    memcpy(m_spendSecretKey, key, sizeof(m_spendSecretKey));
    xmrig::derive_view_secret_key(m_spendSecretKey, m_viewSecretKey);
    xmrig::secret_key_to_public_key(m_spendSecretKey, m_spendPublicKey);
    xmrig::secret_key_to_public_key(m_viewSecretKey, m_viewPublicKey);
}


void xmrig::Job::setMinerTx(const uint8_t* begin, const uint8_t* end, size_t minerTxEphPubKeyOffset, size_t minerTxPubKeyOffset, const Buffer& minerTxMerkleTreeBranch)
{
    m_minerTxPrefix.assign(begin, end);
    m_minerTxEphPubKeyOffset = minerTxEphPubKeyOffset;
    m_minerTxPubKeyOffset = minerTxPubKeyOffset;
    m_minerTxMerkleTreeBranch = minerTxMerkleTreeBranch;
}


void xmrig::Job::generateHashingBlob(String& blob, String& signatureData) const
{
    uint8_t* eph_public_key = m_minerTxPrefix.data() + m_minerTxEphPubKeyOffset;
    uint8_t* txkey_pub = m_minerTxPrefix.data() + m_minerTxPubKeyOffset;

    uint8_t txkey_sec[32];

    generate_keys(txkey_pub, txkey_sec);

    uint8_t derivation[32];

    generate_key_derivation(m_viewPublicKey, txkey_sec, derivation);
    derive_public_key(derivation, 0, m_spendPublicKey, eph_public_key);

    uint8_t buf[32 * 3] = {};
    memcpy(buf, txkey_pub, 32);
    memcpy(buf + 32, eph_public_key, 32);

    generate_key_derivation(txkey_pub, m_viewSecretKey, derivation);
    derive_secret_key(derivation, 0, m_spendSecretKey, buf + 64);

    signatureData = xmrig::Cvt::toHex(buf, sizeof(buf));

    uint8_t root_hash[32];
    const uint8_t* p = m_minerTxPrefix.data();
    xmrig::BlockTemplate::CalculateRootHash(p, p + m_minerTxPrefix.size(), m_minerTxMerkleTreeBranch, root_hash);

    blob = rawBlob();
    const uint64_t offset = nonceOffset() + nonceSize() + BlockTemplate::SIGNATURE_SIZE + 2 /* vote */;
    xmrig::Cvt::toHex(blob.data() + offset * 2, 64, root_hash, 32);
}


#else


void xmrig::Job::generateMinerSignature(const uint8_t* blob, size_t size, uint8_t* out_sig) const
{
    uint8_t tmp[kMaxBlobSize];
    memcpy(tmp, blob, size);

    // Fill signature with zeros
    memset(tmp + nonceOffset() + nonceSize(), 0, BlockTemplate::SIGNATURE_SIZE);

    uint8_t prefix_hash[32];
    xmrig::keccak(tmp, static_cast<int>(size), prefix_hash, sizeof(prefix_hash));
    xmrig::generate_signature(prefix_hash, m_ephPublicKey, m_ephSecretKey, out_sig);
}


#endif
