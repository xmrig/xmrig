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

#ifndef XMRIG_BLOCKTEMPLATE_H
#define XMRIG_BLOCKTEMPLATE_H


#include "base/crypto/Coin.h"
#include "base/tools/Buffer.h"
#include "base/tools/String.h"


namespace xmrig {


struct BlockTemplate
{
    enum {
        HASH_SIZE = 32,
        KEY_SIZE = 32,
        SIGNATURE_SIZE = 64,
        NONCE_SIZE = 4,
    };

    Buffer raw_blob;
    size_t eph_public_key_index;
    size_t tx_pubkey_index;
    uint64_t tx_extra_nonce_size;
    size_t tx_extra_nonce_index;
    size_t miner_tx_prefix_begin_index;
    size_t miner_tx_prefix_end_index;

    // Block header
    uint8_t major_version;
    uint8_t minor_version;
    uint64_t timestamp;
    uint8_t prev_id[HASH_SIZE];
    uint8_t nonce[NONCE_SIZE];

    bool has_miner_signature;
    uint8_t miner_signature[SIGNATURE_SIZE];
    uint8_t vote[2];

    // Miner tx
    uint64_t tx_version;
    uint64_t unlock_time;
    uint64_t num_inputs;
    uint8_t input_type;
    uint64_t height;
    uint64_t num_outputs;
    uint64_t amount;
    uint8_t output_type;
    uint8_t eph_public_key[KEY_SIZE];
    uint64_t extra_size;
    Buffer extra;
    uint8_t vin_rct_type;

    // Transaction hashes
    uint64_t num_hashes;
    Buffer hashes;

    Buffer miner_tx_merkle_tree_branch;
    uint8_t root_hash[HASH_SIZE];

    Buffer hashingBlob;

    bool Init(const String& blockTemplate, Coin coin);

    static void CalculateMinerTxHash(const uint8_t* prefix_begin, const uint8_t* prefix_end, uint8_t* hash);
    static void CalculateRootHash(const uint8_t* prefix_begin, const uint8_t* prefix_end, const Buffer& miner_tx_merkle_tree_branch, uint8_t* root_hash);
    void CalculateMerkleTreeHash();
    void GenerateHashingBlob();
};


} /* namespace xmrig */


#endif /* XMRIG_BLOCKTEMPLATE_H */
