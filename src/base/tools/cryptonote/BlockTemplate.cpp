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


void xmrig::BlockTemplate::calculateRootHash(const uint8_t *prefix_begin, const uint8_t *prefix_end, const Buffer &miner_tx_merkle_tree_branch, uint32_t miner_tx_merkle_tree_path, uint8_t *root_hash)
{
    calculateMinerTxHash(prefix_begin, prefix_end, root_hash);

    const size_t depth = miner_tx_merkle_tree_branch.size() / kHashSize;

    for (size_t d = 0; d < depth; ++d) {
        uint8_t h[kHashSize * 2];

        const uint32_t t = (miner_tx_merkle_tree_path >> (depth - d - 1)) & 1;

        memcpy(h + kHashSize * t, root_hash, kHashSize);
        memcpy(h + kHashSize * (t ^ 1), miner_tx_merkle_tree_branch.data() + d * kHashSize, kHashSize);

        keccak(h, kHashSize * 2, root_hash, kHashSize);
    }
}


void xmrig::BlockTemplate::calculateMerkleTreeHash(uint32_t index)
{
    m_minerTxMerkleTreeBranch.clear();
	m_minerTxMerkleTreePath = 0;

    const size_t count = m_hashes.size() / kHashSize;
    const uint8_t *h = m_hashes.data();

	if (count == 1) {
        memcpy(m_rootHash, h, kHashSize);
	}
	else if (count == 2) {
        keccak(h, kHashSize * 2, m_rootHash, kHashSize);

		m_minerTxMerkleTreeBranch.reserve(1);
        m_minerTxMerkleTreeBranch.insert(m_minerTxMerkleTreeBranch.end(), h + kHashSize * (index ^ 1), h + kHashSize * ((index ^ 1) + 1));
		m_minerTxMerkleTreePath = static_cast<uint32_t>(index);
	}
	else {
		uint8_t h2[kHashSize];
        memcpy(h2, h + kHashSize * index, kHashSize);

		size_t cnt = 1, proof_max_size = 0;
		do {
			cnt <<= 1;
			++proof_max_size;
		} while (cnt <= count);
		cnt >>= 1;

		m_minerTxMerkleTreeBranch.reserve(proof_max_size);

        Buffer ints(cnt * kHashSize);

		const size_t k = cnt * 2 - count;
		memcpy(ints.data(), h, k * kHashSize);

		for (size_t i = k, j = k; j < cnt; i += 2, ++j) {
			keccak(h + i * kHashSize, kHashSize * 2, ints.data() + j * kHashSize, kHashSize);

			if (memcmp(h + i * kHashSize, h2, kHashSize) == 0) {
                m_minerTxMerkleTreeBranch.insert(m_minerTxMerkleTreeBranch.end(), h + kHashSize * (i + 1), h + kHashSize * (i + 2));
                memcpy(h2, ints.data() + j * kHashSize, kHashSize);
			}
			else if (memcmp(h + (i + 1) * kHashSize, h2, kHashSize) == 0) {
                m_minerTxMerkleTreeBranch.insert(m_minerTxMerkleTreeBranch.end(), h + kHashSize * i, h + kHashSize * (i + 1));
                memcpy(h2, ints.data() + j * kHashSize, kHashSize);
				m_minerTxMerkleTreePath = 1;
			}
		}

		while (cnt >= 2) {
			cnt >>= 1;
			for (size_t i = 0, j = 0; j < cnt; i += 2, ++j) {
                uint8_t tmp[kHashSize];
				keccak(ints.data() + i * kHashSize, kHashSize * 2, tmp, kHashSize);

                if (memcmp(ints.data() + i * kHashSize, h2, kHashSize) == 0) {
                    m_minerTxMerkleTreeBranch.insert(m_minerTxMerkleTreeBranch.end(), ints.data() + kHashSize * (i + 1), ints.data() + kHashSize * (i + 2));
					memcpy(h2, tmp, kHashSize);
					m_minerTxMerkleTreePath <<= 1;
				}
                else if (memcmp(ints.data() + (i + 1) * kHashSize, h2, kHashSize) == 0) {
                    m_minerTxMerkleTreeBranch.insert(m_minerTxMerkleTreeBranch.end(), ints.data() + kHashSize * i, ints.data() + kHashSize * (i + 1));
					memcpy(h2, tmp, kHashSize);
					m_minerTxMerkleTreePath = (m_minerTxMerkleTreePath << 1) | 1;
				}

				memcpy(ints.data() + j * kHashSize, tmp, kHashSize);
			}
		}

		memcpy(m_rootHash, ints.data(), kHashSize);
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

    uint64_t k = m_numHashes + 1;
    while (k >= 0x80) {
        out.emplace_back((static_cast<uint8_t>(k) & 0x7F) | 0x80);
        k >>= 7;
    }
    out.emplace_back(static_cast<uint8_t>(k));
}


bool xmrig::BlockTemplate::parse(bool hashes)
{
    BlobReader<true> ar(m_blob.data(), m_blob.size());

    // Block header
    ar(m_version.first);
    ar(m_version.second);
    ar(m_timestamp);
    ar(m_prevId, kHashSize);

    setOffset(NONCE_OFFSET, ar.index());
    ar.skip(kNonceSize);

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
    else if ((m_outputType != 2) && (m_outputType != 3)) {
        return false;
    }

    setOffset(EPH_PUBLIC_KEY_OFFSET, ar.index());

    ar(m_ephPublicKey, kKeySize);

    if (is_fcmp_pp) {
        ar(m_carrotViewTag);
        ar(m_janusAnchor);
    }

    if (m_coin == Coin::ZEPHYR) {
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

    if (m_coin == Coin::TOWNFORGE) {
      ar(m_unlockTime);
    }

    ar(m_extraSize);

    setOffset(TX_EXTRA_OFFSET, ar.index());

    BlobReader<true> ar_extra(blob(TX_EXTRA_OFFSET), m_extraSize);
    ar.skip(m_extraSize);

    bool pubkey_offset_first = true;

    while (ar_extra.index() < m_extraSize) {
        uint64_t extra_tag  = 0;
        uint64_t size       = 0;

        ar_extra(extra_tag);

        switch (extra_tag) {
        case 0x01: // TX_EXTRA_TAG_PUBKEY
            if (pubkey_offset_first) {
                pubkey_offset_first = false;
                setOffset(TX_PUBKEY_OFFSET, offset(TX_EXTRA_OFFSET) + ar_extra.index());
            }
            ar_extra.skip(kKeySize);
            break;

        case 0x02: // TX_EXTRA_NONCE
            ar_extra(size);
            setOffset(TX_EXTRA_NONCE_OFFSET, offset(TX_EXTRA_OFFSET) + ar_extra.index());
            ar_extra(m_txExtraNonce, size);
            break;

        case 0x03: // TX_EXTRA_MERGE_MINING_TAG
            ar_extra(size);
            setOffset(TX_EXTRA_MERGE_MINING_TAG_OFFSET, offset(TX_EXTRA_OFFSET) + ar_extra.index());
            ar_extra(m_txMergeMiningTag, size);
            break;

        default:
            return false; // TODO(SChernykh): handle other tags
        }
    }

    if (m_coin == Coin::ZEPHYR) {
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

    // Other transaction hashes
    ar(m_numHashes);

    if (hashes) {
        // FCMP++ layout:
        //
        // index 0  fcmp_pp_n_tree_layers + 31 zero bytes
        // index 1  fcmp_pp_tree_root
        // index 2  coinbase transaction hash
        // index 3+ other transaction hashes
        //
        // pre-FCMP++ layout:
        //
        // index 0  coinbase transaction hash
        // index 1+ other transaction hashes
        //
        const uint32_t coinbase_tx_index = is_fcmp_pp ? 2 : 0;

        m_hashes.clear();
        m_hashes.resize((coinbase_tx_index + m_numHashes + 1) * kHashSize);

        uint8_t* data = m_hashes.data() + coinbase_tx_index * kHashSize;

        calculateMinerTxHash(blob(MINER_TX_PREFIX_OFFSET), blob(MINER_TX_PREFIX_END_OFFSET), data);

        for (uint64_t i = 1; i <= m_numHashes; ++i) {
            Span h;
            ar(h, kHashSize);
            memcpy(data + i * kHashSize, h.data(), kHashSize);
        }

        if (is_fcmp_pp) {
            ar(m_FCMPTreeLayers);
            ar(m_FCMPTreeRoot);

            m_hashes[0] = m_FCMPTreeLayers;
            memcpy(m_hashes.data() + kHashSize, m_FCMPTreeRoot, kHashSize);
        }

        calculateMerkleTreeHash(coinbase_tx_index);
    }

    return true;
}
