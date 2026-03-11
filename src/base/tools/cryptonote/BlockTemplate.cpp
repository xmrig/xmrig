/* XMRig
 * Copyright (c) 2012-2013 The Cryptonote developers
 * Copyright (c) 2014-2021 The Monero Project
 * Copyright (c) 2018-2023 SChernykh   <https://github.com/SChernykh>
 * Copyright (c) 2016-2023 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
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

#include "base/tools/cryptonote/BlockTemplate.h"
#include "3rdparty/rapidjson/document.h"
#include "base/crypto/keccak.h"
#include "base/tools/cryptonote/BlobReader.h"
#include "base/tools/Cvt.h"

namespace
{
    constexpr uint8_t kSalviumProtocolHF              = 2;
    constexpr uint8_t kSalviumProtocolTxVersionLegacy = 2;
    constexpr uint8_t kSalviumProtocolTxVersionCarrot = 4;
    constexpr uint8_t kSalviumProtocolTxVersionTokens = 5;
    constexpr uint8_t kSalviumCarrotHF                = 10;
    constexpr uint8_t kSalviumTokensHF                = 11;
    constexpr uint8_t kTxOutToKey               = 2;
    constexpr uint8_t kTxOutToTaggedKey         = 3;
    constexpr uint8_t kTxOutToCarrotV1          = 4;
}

void xmrig::BlockTemplate::calculateMinerTxHash(const uint8_t *prefix_begin, const uint8_t *prefix_end, uint8_t *hash)
{
    uint8_t hashes[kHashSize * 3];

    // Calculate 3 partial hashes

    // 1. Prefix
    keccak(prefix_begin, static_cast<int>(prefix_end - prefix_begin), hashes, kHashSize);

    // 2. Base RCT, single 0 byte in miner tx
    static const uint8_t known_second_hash[kHashSize] = {
        188,54,120,158,122,30,40,20,54,70,66,41,130,143,129,125,102,18,247,180,119,214,101,145,255,150,169,224,100,188,201,138
    };
    memcpy(hashes + kHashSize, known_second_hash, kHashSize);

    // 3. Prunable RCT, empty in miner tx
    memset(hashes + kHashSize * 2, 0, kHashSize);

    // Calculate miner transaction hash
    keccak(hashes, sizeof(hashes), hash, kHashSize);
}


void xmrig::BlockTemplate::calculateRootHash(const uint8_t *prefix_begin, const uint8_t *prefix_end, const Buffer &miner_tx_merkle_tree_branch, uint8_t *root_hash)
{
    calculateMinerTxHash(prefix_begin, prefix_end, root_hash);

    for (size_t i = 0; i < miner_tx_merkle_tree_branch.size(); i += kHashSize) {
        uint8_t h[kHashSize * 2];

        memcpy(h, root_hash, kHashSize);
        memcpy(h + kHashSize, miner_tx_merkle_tree_branch.data() + i, kHashSize);

        keccak(h, kHashSize * 2, root_hash, kHashSize);
    }
}


void xmrig::BlockTemplate::calculateMerkleTreeHash()
{
    m_minerTxMerkleTreeBranch.clear();

    const uint64_t count = m_numHashes + baseTransactionCount();
    const uint8_t *h = m_hashes.data();

    if (count == 1) {
        memcpy(m_rootHash, h, kHashSize);
    }
    else if (count == 2) {
        m_minerTxMerkleTreeBranch.insert(m_minerTxMerkleTreeBranch.end(), h + kHashSize, h + kHashSize * 2);
        keccak(h, kHashSize * 2, m_rootHash, kHashSize);
    }
    else {
        size_t i    = 0;
        size_t j    = 0;
        size_t cnt  = 0;

        for (i = 0, cnt = 1; cnt <= count; ++i, cnt <<= 1) {}

        cnt >>= 1;

        m_minerTxMerkleTreeBranch.reserve(kHashSize * (i - 1));

        Buffer ints(cnt * kHashSize);
        memcpy(ints.data(), h, (cnt * 2 - count) * kHashSize);

        for (i = cnt * 2 - count, j = cnt * 2 - count; j < cnt; i += 2, ++j) {
            if (i == 0) {
                m_minerTxMerkleTreeBranch.insert(m_minerTxMerkleTreeBranch.end(), h + kHashSize, h + kHashSize * 2);
            }
            keccak(h + i * kHashSize, kHashSize * 2, ints.data() + j * kHashSize, kHashSize);
        }

        while (cnt > 2) {
            cnt >>= 1;
            for (i = 0, j = 0; j < cnt; i += 2, ++j) {
                if (i == 0) {
                    m_minerTxMerkleTreeBranch.insert(m_minerTxMerkleTreeBranch.end(), ints.data() + kHashSize, ints.data() + kHashSize * 2);
                }
                keccak(ints.data() + i * kHashSize, kHashSize * 2, ints.data() + j * kHashSize, kHashSize);
            }
        }

        m_minerTxMerkleTreeBranch.insert(m_minerTxMerkleTreeBranch.end(), ints.data() + kHashSize, ints.data() + kHashSize * 2);
        keccak(ints.data(), kHashSize * 2, m_rootHash, kHashSize);
    }
}


bool xmrig::BlockTemplate::parse(const Buffer &blocktemplate, const Coin &coin, bool hashes)
{
    if (blocktemplate.size() < kMinSize) {
        return false;
    }

    m_blob  = blocktemplate;
    m_coin  = coin;
    bool rc = false;

    try {
        rc = parse(hashes);
    } catch (...) {}

    return rc;
}


bool xmrig::BlockTemplate::parse(const char *blocktemplate, size_t size, const Coin &coin, bool hashes)
{
    if (size < (kMinSize * 2) || !Cvt::fromHex(m_blob, blocktemplate, size)) {
        return false;
    }

    m_coin  = coin;
    bool rc = false;

    try {
        rc = parse(hashes);
    } catch (...) {}

    return rc;
}


bool xmrig::BlockTemplate::parse(const rapidjson::Value &blocktemplate, const Coin &coin, bool hashes)
{
    return blocktemplate.IsString() && parse(blocktemplate.GetString(), blocktemplate.GetStringLength(), coin, hashes);
}


bool xmrig::BlockTemplate::parse(const String &blocktemplate, const Coin &coin, bool hashes)
{
    return parse(blocktemplate.data(), blocktemplate.size(), coin, hashes);
}


void xmrig::BlockTemplate::generateHashingBlob(Buffer &out) const
{
    out.clear();
    out.reserve(offset(MINER_TX_PREFIX_OFFSET) + kHashSize + 3);

    out.assign(m_blob.begin(), m_blob.begin() + offset(MINER_TX_PREFIX_OFFSET));
    out.insert(out.end(), m_rootHash, m_rootHash + kHashSize);

    uint64_t k = m_numHashes + baseTransactionCount();
    while (k >= 0x80) {
        out.emplace_back((static_cast<uint8_t>(k) & 0x7F) | 0x80);
        k >>= 7;
    }
    out.emplace_back(static_cast<uint8_t>(k));
}


bool xmrig::BlockTemplate::parse(bool hashes)
{
    BlobReader<true> ar(m_blob.data(), m_blob.size());
    m_hasProtocolTx = false;

    // Block header
    ar(m_version.first);
    ar(m_version.second);
    ar(m_timestamp);
    ar(m_prevId, kHashSize);

    setOffset(NONCE_OFFSET, ar.index());
    ar.skip(kNonceSize);

    if (m_coin == Coin::SALVIUM && m_version.first >= kSalviumProtocolHF) {
        m_hasProtocolTx = true;
    }

    // Wownero block template has miner signature starting from version 18
    if (m_coin == Coin::WOWNERO && majorVersion() >= 18) {
        ar(m_minerSignature, kSignatureSize);
        ar(m_vote);
    }

    if (m_coin == Coin::ZEPHYR) {
        uint8_t pricing_record[120];
        ar(pricing_record);
    }

    // Miner transaction begin
    // Prefix begin
    setOffset(MINER_TX_PREFIX_OFFSET, ar.index());

    ar(m_txVersion);

    if (m_coin != Coin::TOWNFORGE) {
      ar(m_unlockTime);
    }

    ar(m_numInputs);

    // must be 1 input
    if (m_numInputs != 1) {
        return false;
    }

    ar(m_inputType);

    // input type must be txin_gen (0xFF)
    if (m_inputType != 0xFF) {
        return false;
    }

    ar(m_height);
    ar(m_numOutputs);

    if (m_coin == Coin::ZEPHYR) {
        if (m_numOutputs < 2) {
            return false;
        }
    }
    else if (m_coin == Coin::SALVIUM) {
        if (m_numOutputs < 1) {
            return false;
        }
    }
    else if (m_numOutputs != 1) {
        return false;
    }

    ar(m_amount);
    ar(m_outputType);

    const bool is_fcmp_pp = (m_coin == Coin::MONERO) && (m_version.first >= 17);

    // output type must be txout_to_key (2) or txout_to_tagged_key (3) for versions < 17, and txout_to_carrot_v1 (0) for version FCMP++
    if (is_fcmp_pp && (m_outputType == 0)) {
        // all good
    }
    else if (m_coin == Coin::SALVIUM) {
        if ((m_outputType != kTxOutToKey) && (m_outputType != kTxOutToTaggedKey) && (m_outputType != kTxOutToCarrotV1)) {
            return false;
        }
    }
    else if ((m_outputType != kTxOutToKey) && (m_outputType != kTxOutToTaggedKey)) {
        return false;
    }

    setOffset(EPH_PUBLIC_KEY_OFFSET, ar.index());

    ar(m_ephPublicKey, kKeySize);

    if (is_fcmp_pp) {
        ar(m_carrotViewTag);
        ar(m_janusAnchor);
    }
    else if (m_coin == Coin::SALVIUM) {
        if (!parseSalviumOutput(ar, m_outputType, true)) {
            return false;
        }
    }
    else if (m_coin == Coin::ZEPHYR) {
        if (m_outputType != 2) {
            return false;
        }

        uint64_t asset_type_len;
        ar(asset_type_len);
        ar.skip(asset_type_len);
        ar(m_viewTag);

        for (uint64_t k = 1; k < m_numOutputs; ++k) {
            uint64_t amount2;
            ar(amount2);

            uint8_t output_type2;
            ar(output_type2);
            if (output_type2 != 2) {
                return false;
            }

            Span key2;
            ar(key2, kKeySize);

            ar(asset_type_len);
            ar.skip(asset_type_len);

            uint8_t view_tag2;
            ar(view_tag2);
        }
    }
    else if (m_outputType == 3) {
        ar(m_viewTag);
    }

    if (m_coin == Coin::SALVIUM && m_numOutputs > 1) {
        for (uint64_t k = 1; k < m_numOutputs; ++k) {
            uint64_t amount2;
            ar(amount2);

            uint8_t output_type2;
            ar(output_type2);

            if ((output_type2 != kTxOutToKey) && (output_type2 != kTxOutToTaggedKey) && (output_type2 != kTxOutToCarrotV1)) {
                return false;
            }

            Span key2;
            ar(key2, kKeySize);

            if (!parseSalviumOutput(ar, output_type2, false)) {
                return false;
            }
        }
    }

    if (m_coin == Coin::TOWNFORGE) {
      ar(m_unlockTime);
    }

    ar(m_extraSize);

    setOffset(TX_EXTRA_OFFSET, ar.index());

    BlobReader<true> ar_extra(blob(TX_EXTRA_OFFSET), m_extraSize);
    const uint8_t *extra_ptr = blob(TX_EXTRA_OFFSET);
    ar.skip(m_extraSize);

    bool pubkey_offset_first = true;

    while (ar_extra.index() < m_extraSize) {
        uint64_t extra_tag  = 0;
        uint64_t size       = 0;

        ar_extra(extra_tag);

        switch (extra_tag) {
        case 0x00: // TX_EXTRA_TAG_PADDING
            while (ar_extra.index() < m_extraSize && extra_ptr[ar_extra.index()] == 0) {
                ar_extra.skip(1);
            }
            break;

        case 0x01: // TX_EXTRA_TAG_PUBKEY
            if (pubkey_offset_first) {
                pubkey_offset_first = false;
                setOffset(TX_PUBKEY_OFFSET, offset(TX_EXTRA_OFFSET) + ar_extra.index());
            }
            ar_extra.skip(kKeySize);
            break;

        case 0x02: // TX_EXTRA_NONCE
            if (!ar_extra(size)) {
                return false;
            }
            setOffset(TX_EXTRA_NONCE_OFFSET, offset(TX_EXTRA_OFFSET) + ar_extra.index());
            ar_extra(m_txExtraNonce, size);
            break;

        case 0x03: // TX_EXTRA_MERGE_MINING_TAG
            if (!ar_extra(size)) {
                return false;
            }
            setOffset(TX_EXTRA_MERGE_MINING_TAG_OFFSET, offset(TX_EXTRA_OFFSET) + ar_extra.index());
            ar_extra(m_txMergeMiningTag, size);
            break;

        case 0x04: // TX_EXTRA_TAG_ADDITIONAL_PUBKEYS
            if (!ar_extra(size)) {
                return false;
            }
            ar_extra.skip(kKeySize * size);
            break;

        default:
            if (!ar_extra(size)) {
                return false;
            }
            if (size > ar_extra.remaining()) {
                return false;
            }
            ar_extra.skip(static_cast<size_t>(size));
        }
    }

    if (m_coin == Coin::SALVIUM) {

      uint8_t tx_type;
      ar(tx_type);

      if (tx_type != 1) {
        return false;
      }

      uint64_t amount_burnt;
      ar(amount_burnt);

    } else if (m_coin == Coin::ZEPHYR) {
        uint64_t pricing_record_height, amount_burnt, amount_minted;
        ar(pricing_record_height);
        ar(amount_burnt);
        ar(amount_minted);
    }

    setOffset(MINER_TX_PREFIX_END_OFFSET, ar.index());
    // Prefix end

    // RCT signatures (empty in miner transaction)
    uint8_t vin_rct_type = 0;
    ar(vin_rct_type);

    // no way I'm parsing a full game update here
    if (m_coin == Coin::TOWNFORGE && m_height % 720 == 0) {
      return true;
    }

    // must be RCTTypeNull (0)
    if (vin_rct_type != 0) {
        return false;
    }

    const size_t miner_tx_end = ar.index();
    // Miner transaction end

    // Miner transaction must have exactly 1 byte with value 0 after the prefix
    if ((miner_tx_end != offset(MINER_TX_PREFIX_END_OFFSET) + 1) || (*blob(MINER_TX_PREFIX_END_OFFSET) != 0)) {
        return false;
    }

    if (m_hasProtocolTx) {

      // Protocol transaction begin
      // Prefix begin
      setOffset(PROTOCOL_TX_PREFIX_OFFSET, ar.index());

      // Parse/skip the protocol_tx
      uint64_t protocol_tx_version, protocol_tx_unlock_time, protocol_tx_num_inputs;
      ar(protocol_tx_version);
      const uint8_t expected_protocol_version =
        (majorVersion() >= kSalviumTokensHF) ? kSalviumProtocolTxVersionTokens :
        (majorVersion() >= kSalviumCarrotHF) ? kSalviumProtocolTxVersionCarrot : kSalviumProtocolTxVersionLegacy;
      if (protocol_tx_version != expected_protocol_version) {
        return false;
      }

      ar(protocol_tx_unlock_time);
      ar(protocol_tx_num_inputs);

      // must be 1 input
      if (protocol_tx_num_inputs != 1) {
        return false;
      }

      uint8_t protocol_input_type;
      ar(protocol_input_type);

      // input type must be txin_gen (0xFF)
      if (protocol_input_type != 0xFF) {
        return false;
      }

      uint64_t protocol_height;
      ar(protocol_height);

      // height must match miner_tx
      if (protocol_height != m_height) {
        return false;
      }

      uint64_t protocol_num_outputs;
      ar(protocol_num_outputs);
      
      for (size_t protocol_output_idx=0; protocol_output_idx<protocol_num_outputs; ++protocol_output_idx) {

        uint64_t out_amount;
        ar(out_amount);
        
        uint8_t out_type;
        ar(out_type);
        
        // output type must be txout_to_key (2), txout_to_tagged_key (3) or txout_to_carrot_v1 (4)
        if ((out_type != kTxOutToKey) && (out_type != kTxOutToTaggedKey) && (out_type != kTxOutToCarrotV1)) {
          return false;
        }
        
        Span out_pubkey;
        ar(out_pubkey, kKeySize);

        if (!parseSalviumOutput(ar, out_type, false)) {
          return false;
        }
      }

      uint64_t protocol_extra_size;
      ar(protocol_extra_size);
      ar.skip(protocol_extra_size);

      uint8_t protocol_tx_type;
      ar(protocol_tx_type);
      if (protocol_tx_type != 2) {
        return false;
      }

      setOffset(PROTOCOL_TX_PREFIX_END_OFFSET, ar.index());
      // Prefix end

      // RCT signatures (empty in protocol transaction)
      uint8_t protocol_rct_type = 0;
      ar(protocol_rct_type);

      // must be RCTTypeNull (0)
      if (protocol_rct_type != 0) {
        return false;
      }

      const size_t protocol_tx_end = ar.index();
      // Protocol transaction end

      // Protocol transaction must have exactly 1 byte with value 0 after the prefix
      if ((protocol_tx_end != offset(PROTOCOL_TX_PREFIX_END_OFFSET) + 1) || (*blob(PROTOCOL_TX_PREFIX_END_OFFSET) != 0)) {
        return false;
      }
    }

    // Other transaction hashes
    ar(m_numHashes);

    if (hashes) {
        const uint64_t base_count = baseTransactionCount();
        m_hashes.resize((m_numHashes + base_count) * kHashSize);

        // Miner TX hash always goes first
        calculateMinerTxHash(blob(MINER_TX_PREFIX_OFFSET), blob(MINER_TX_PREFIX_END_OFFSET), m_hashes.data());

        uint64_t offset = 1;

        if (m_hasProtocolTx) {
            calculateMinerTxHash(blob(PROTOCOL_TX_PREFIX_OFFSET), blob(PROTOCOL_TX_PREFIX_END_OFFSET), m_hashes.data() + offset * kHashSize);
            ++offset;
        }

        for (uint64_t i = 0; i < m_numHashes; ++i) {
            Span h;
            ar(h, kHashSize);
            memcpy(m_hashes.data() + (offset + i) * kHashSize, h.data(), kHashSize);
        }

        calculateMerkleTreeHash();
    }

    return true;
}


bool xmrig::BlockTemplate::parseSalviumOutput(BlobReader<true> &ar, uint8_t outputType, bool storeExtraData)
{
    uint64_t asset_type_len = 0;
    ar(asset_type_len);

    if (asset_type_len > ar.remaining()) {
        return false;
    }

    if (asset_type_len > 0 && !ar.skip(static_cast<size_t>(asset_type_len))) {
        return false;
    }

    if (outputType == kTxOutToKey || outputType == kTxOutToTaggedKey) {
        uint64_t unlock_time = 0;
        ar(unlock_time);
    }

    if (outputType == kTxOutToTaggedKey) {
        uint8_t view_tag = 0;
        ar(view_tag);

        if (storeExtraData) {
            m_viewTag = view_tag;
        }
    }
    else if (outputType == kTxOutToCarrotV1) {
        uint8_t carrot_view_tag[kCarrotViewTagSize];
        uint8_t carrot_anchor[kCarrotAnchorSize];

        ar(carrot_view_tag);
        ar(carrot_anchor);

        if (storeExtraData) {
            memcpy(m_carrotViewTag, carrot_view_tag, kCarrotViewTagSize);
            memcpy(m_janusAnchor, carrot_anchor, kCarrotAnchorSize);
        }
    }

    return true;
}
