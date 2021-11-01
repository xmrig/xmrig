/* XMRig
 * Copyright (c) 2012-2013 The Cryptonote developers
 * Copyright (c) 2014-2021 The Monero Project
 * Copyright (c) 2018-2021 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2021 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#ifndef XMRIG_BLOCKTEMPLATE_H
#define XMRIG_BLOCKTEMPLATE_H


#include "3rdparty/rapidjson/fwd.h"
#include "base/crypto/Coin.h"
#include "base/tools/Buffer.h"
#include "base/tools/String.h"
#include "base/tools/Span.h"


namespace xmrig {


class BlockTemplate
{
public:
    static constexpr size_t kHashSize       = 32;
    static constexpr size_t kKeySize        = 32;
    static constexpr size_t kNonceSize      = 4;
    static constexpr size_t kSignatureSize  = 64;

#   ifdef XMRIG_PROXY_PROJECT
    static constexpr bool kCalcHashes       = true;
#   else
    static constexpr bool kCalcHashes       = false;
#   endif

    enum Offset : uint32_t {
        NONCE_OFFSET,
        MINER_TX_PREFIX_OFFSET,
        MINER_TX_PREFIX_END_OFFSET,
        EPH_PUBLIC_KEY_OFFSET,
        TX_EXTRA_OFFSET,
        TX_PUBKEY_OFFSET,
        TX_EXTRA_NONCE_OFFSET,
        TX_EXTRA_MERGE_MINING_TAG_OFFSET,
        OFFSET_COUNT
    };

    inline const Coin &coin() const                         { return m_coin; }
    inline const uint8_t *blob() const                      { return m_blob.data(); }
    inline const uint8_t *blob(Offset offset) const         { return m_blob.data() + m_offsets[offset]; }
    inline size_t offset(Offset offset) const               { return m_offsets[offset]; }
    inline size_t size() const                              { return m_blob.size(); }

    // Block header
    inline uint8_t majorVersion() const                     { return m_version.first; }
    inline uint8_t minorVersion() const                     { return m_version.second; }
    inline uint64_t timestamp() const                       { return m_timestamp; }
    inline const Span &prevId() const                       { return m_prevId; }
    inline const uint8_t *nonce() const                     { return blob(NONCE_OFFSET); }

    // Wownero miner signature
    inline bool hasMinerSignature() const                   { return !m_minerSignature.empty(); }
    inline const Span &minerSignature() const               { return m_minerSignature; }
    inline const uint8_t *vote() const                      { return m_vote; }

    // Miner tx
    inline uint64_t txVersion() const                       { return m_txVersion; }
    inline uint64_t unlockTime() const                      { return m_unlockTime; }
    inline uint64_t numInputs() const                       { return m_numInputs; }
    inline uint8_t inputType() const                        { return m_inputType; }
    inline uint64_t height() const                          { return m_height; }
    inline uint64_t numOutputs() const                      { return m_numOutputs; }
    inline uint64_t amount() const                          { return m_amount; }
    inline uint64_t outputType() const                      { return m_outputType; }
    inline const Span &ephPublicKey() const                 { return m_ephPublicKey; }
    inline const Span &txExtraNonce() const                 { return m_txExtraNonce; }
    inline const Span &txMergeMiningTag() const             { return m_txMergeMiningTag; }

    // Transaction hashes
    inline uint64_t numHashes() const                       { return m_numHashes; }
    inline const Buffer &hashes() const                     { return m_hashes; }
    inline const Buffer &minerTxMerkleTreeBranch() const    { return m_minerTxMerkleTreeBranch; }
    inline const uint8_t *rootHash() const                  { return m_rootHash; }

    inline Buffer generateHashingBlob() const
    {
        Buffer out;
        generateHashingBlob(out);

        return out;
    }

    static void calculateMinerTxHash(const uint8_t *prefix_begin, const uint8_t *prefix_end, uint8_t *hash);
    static void calculateRootHash(const uint8_t *prefix_begin, const uint8_t *prefix_end, const Buffer &miner_tx_merkle_tree_branch, uint8_t *root_hash);

    bool parse(const Buffer &blocktemplate, const Coin &coin, bool hashes = kCalcHashes);
    bool parse(const char *blocktemplate, size_t size, const Coin &coin, bool hashes);
    bool parse(const rapidjson::Value &blocktemplate, const Coin &coin, bool hashes = kCalcHashes);
    bool parse(const String &blocktemplate, const Coin &coin, bool hashes = kCalcHashes);
    void calculateMerkleTreeHash();
    void generateHashingBlob(Buffer &out) const;

private:
    static constexpr size_t kMinSize = 76;

    inline void setOffset(Offset offset, size_t value)  { m_offsets[offset] = static_cast<uint32_t>(value); }

    bool parse(bool hashes);

    Buffer m_blob;
    Coin m_coin;
    uint32_t m_offsets[OFFSET_COUNT]{};

    std::pair<uint8_t, uint8_t> m_version;
    uint64_t m_timestamp    = 0;
    Span m_prevId;

    Span m_minerSignature;
    uint8_t m_vote[2]{};

    uint64_t m_txVersion    = 0;
    uint64_t m_unlockTime   = 0;
    uint64_t m_numInputs    = 0;
    uint8_t m_inputType     = 0;
    uint64_t m_height       = 0;
    uint64_t m_numOutputs   = 0;
    uint64_t m_amount       = 0;
    uint8_t m_outputType    = 0;
    Span m_ephPublicKey;
    uint64_t m_extraSize    = 0;
    Span m_txExtraNonce;
    Span m_txMergeMiningTag = 0;
    uint64_t m_numHashes    = 0;
    Buffer m_hashes;
    Buffer m_minerTxMerkleTreeBranch;
    uint8_t m_rootHash[kHashSize]{};
};


} /* namespace xmrig */


#endif /* XMRIG_BLOCKTEMPLATE_H */
