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
#include "base/tools/Cvt.h"
#include "base/tools/cryptonote/BlobReader.h"
#include "base/tools/cryptonote/BlockTemplate.h"


namespace xmrig {


bool BlockTemplate::Init(const String& blockTemplate, Coin coin)
{
    raw_blob = Cvt::fromHex(blockTemplate);

    CBlobReader ar(raw_blob.data(), raw_blob.size());

    // Block header
    ar(major_version);
    ar(minor_version);
    ar(timestamp);
    ar(prev_id);
    ar(nonce);

    // Wownero block template has miner signature starting from version 18
    has_miner_signature = (coin == Coin::WOWNERO) && (major_version >= 18);
    if (has_miner_signature) {
        ar(miner_signature);
        ar(vote);
    }

    // Miner transaction begin
    // Prefix begin
    miner_tx_prefix_begin_index = ar.index();

    ar(tx_version);
    ar(unlock_time);
    ar(num_inputs);

    // must be 1 input
    if (num_inputs != 1)
        return false;

    ar(input_type);

    // input type must be txin_gen (0xFF)
    if (input_type != 0xFF)
        return false;

    ar(height);

    ar(num_outputs);

    // must be 1 output
    if (num_outputs != 1)
        return false;

    ar(amount);
    ar(output_type);

    // output type must be txout_to_key (2)
    if (output_type != 2)
        return false;

    eph_public_key_index = ar.index();

    ar(eph_public_key);
    ar(extra_size);

    const uint64_t tx_extra_index = ar.index();

    ar.readItems(extra, extra_size);

    CBlobReader ar_extra(extra.data(), extra_size);

    tx_extra_nonce_size = 0;
    tx_extra_nonce_index = 0;

    while (ar_extra.index() < extra_size) {
        uint64_t extra_tag = 0;
        ar_extra(extra_tag);

        switch (extra_tag) {
        case 0x01: // TX_EXTRA_TAG_PUBKEY
            tx_pubkey_index = tx_extra_index + ar_extra.index();
            ar_extra.skip(KEY_SIZE);
            break;

        case 0x02: // TX_EXTRA_NONCE
            ar_extra(tx_extra_nonce_size);
            tx_extra_nonce_index = tx_extra_index + ar_extra.index();
            ar_extra.skip(tx_extra_nonce_size);
            break;

        default:
            return false; // TODO: handle other tags
        }
    }

    miner_tx_prefix_end_index = ar.index();
    // Prefix end

    // RCT signatures (empty in miner transaction)
    ar(vin_rct_type);

    // must be RCTTypeNull (0)
    if (vin_rct_type != 0)
        return false;

    const size_t miner_tx_end = ar.index();
    // Miner transaction end

    // Miner transaction must have exactly 1 byte with value 0 after the prefix
    if ((miner_tx_end != miner_tx_prefix_end_index + 1) || (raw_blob[miner_tx_prefix_end_index] != 0))
        return false;

    // Other transaction hashes
    ar(num_hashes);

#   ifdef XMRIG_PROXY_PROJECT
    hashes.resize((num_hashes + 1) * HASH_SIZE);
    CalculateMinerTxHash(raw_blob.data() + miner_tx_prefix_begin_index, raw_blob.data() + miner_tx_prefix_end_index, hashes.data());

    for (uint64_t i = 1; i <= num_hashes; ++i) {
        uint8_t h[HASH_SIZE];
        ar(h);
        memcpy(hashes.data() + i * HASH_SIZE, h, HASH_SIZE);
    }

    CalculateMerkleTreeHash();
#   endif

    return true;
}


void BlockTemplate::CalculateMinerTxHash(const uint8_t* prefix_begin, const uint8_t* prefix_end, uint8_t* hash)
{
    uint8_t hashes[HASH_SIZE * 3];

    // Calculate 3 partial hashes

    // 1. Prefix
    keccak(prefix_begin, static_cast<int>(prefix_end - prefix_begin), hashes, HASH_SIZE);

    // 2. Base RCT, single 0 byte in miner tx
    static const uint8_t known_second_hash[HASH_SIZE] = {
        188,54,120,158,122,30,40,20,54,70,66,41,130,143,129,125,102,18,247,180,119,214,101,145,255,150,169,224,100,188,201,138
    };
    memcpy(hashes + HASH_SIZE, known_second_hash, HASH_SIZE);

    // 3. Prunable RCT, empty in miner tx
    memset(hashes + HASH_SIZE * 2, 0, HASH_SIZE);

    // Calculate miner transaction hash
    keccak(hashes, sizeof(hashes), hash, HASH_SIZE);
}


void BlockTemplate::CalculateMerkleTreeHash()
{
    miner_tx_merkle_tree_branch.clear();

    const uint64_t count = num_hashes + 1;
    uint8_t* h = hashes.data();

    if (count == 1) {
        memcpy(root_hash, h, HASH_SIZE);
    }
    else if (count == 2) {
        miner_tx_merkle_tree_branch.insert(miner_tx_merkle_tree_branch.end(), h + HASH_SIZE, h + HASH_SIZE * 2);
        keccak(h, HASH_SIZE * 2, root_hash, HASH_SIZE);
    }
    else {
        size_t i, j, cnt;

        for (i = 0, cnt = 1; cnt <= count; ++i, cnt <<= 1) {}

        cnt >>= 1;

        miner_tx_merkle_tree_branch.reserve(HASH_SIZE * (i - 1));

        Buffer ints(cnt * HASH_SIZE);
        memcpy(ints.data(), h, (cnt * 2 - count) * HASH_SIZE);

        for (i = cnt * 2 - count, j = cnt * 2 - count; j < cnt; i += 2, ++j) {
            if (i == 0) {
                miner_tx_merkle_tree_branch.insert(miner_tx_merkle_tree_branch.end(), h + HASH_SIZE, h + HASH_SIZE * 2);
            }
            keccak(h + i * HASH_SIZE, HASH_SIZE * 2, ints.data() + j * HASH_SIZE, HASH_SIZE);
        }

        while (cnt > 2) {
            cnt >>= 1;
            for (i = 0, j = 0; j < cnt; i += 2, ++j) {
                if (i == 0) {
                    miner_tx_merkle_tree_branch.insert(miner_tx_merkle_tree_branch.end(), ints.data() + HASH_SIZE, ints.data() + HASH_SIZE * 2);
                }
                keccak(ints.data() + i * HASH_SIZE, HASH_SIZE * 2, ints.data() + j * HASH_SIZE, HASH_SIZE);
            }
        }

        miner_tx_merkle_tree_branch.insert(miner_tx_merkle_tree_branch.end(), ints.data() + HASH_SIZE, ints.data() + HASH_SIZE * 2);
        keccak(ints.data(), HASH_SIZE * 2, root_hash, HASH_SIZE);
    }
}


void BlockTemplate::CalculateRootHash(const uint8_t* prefix_begin, const uint8_t* prefix_end, const Buffer& miner_tx_merkle_tree_branch, uint8_t* root_hash)
{
    CalculateMinerTxHash(prefix_begin, prefix_end, root_hash);

    for (size_t i = 0; i < miner_tx_merkle_tree_branch.size(); i += HASH_SIZE) {
        uint8_t h[HASH_SIZE * 2];

        memcpy(h, root_hash, HASH_SIZE);
        memcpy(h + HASH_SIZE, miner_tx_merkle_tree_branch.data() + i, HASH_SIZE);

        keccak(h, HASH_SIZE * 2, root_hash, HASH_SIZE);
    }
}


void BlockTemplate::GenerateHashingBlob()
{
    hashingBlob.clear();
    hashingBlob.reserve(miner_tx_prefix_begin_index + HASH_SIZE + 3);

    hashingBlob.assign(raw_blob.begin(), raw_blob.begin() + miner_tx_prefix_begin_index);
    hashingBlob.insert(hashingBlob.end(), root_hash, root_hash + HASH_SIZE);

    uint64_t k = num_hashes + 1;
    while (k >= 0x80) {
        hashingBlob.emplace_back((static_cast<uint8_t>(k) & 0x7F) | 0x80);
        k >>= 7;
    }
    hashingBlob.emplace_back(static_cast<uint8_t>(k));
}


} /* namespace xmrig */
