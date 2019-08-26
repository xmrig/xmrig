//
// Created by Haifa Bogdan Adnan on 05/08/2018.
//

#include "../../common/common.h"
#include "../../crypt/base64.h"
#include "../../crypt/hex.h"
#include "../../crypt/random_generator.h"

#include "blake2/blake2.h"
#include "../../common/DLLExport.h"
#include "../../../Argon2_constants.h"
#include "Argon2.h"
#include "Defs.h"

Argon2::Argon2(argon2BlocksPrehash prehash, argon2BlocksFillerPtr filler, argon2BlocksPosthash posthash, void *memory, void *userData) {
    m_prehash = prehash;
    m_filler = filler;
    m_posthash = posthash;
    m_outputMemory = m_seedMemory = (uint8_t*)memory;
    m_userData = userData;
    m_threads = 1;
}

int Argon2::generateHashes(const Argon2Profile &profile, HashData &hashData) {
    if(initializeSeeds(profile, hashData)) {
        if(fillBlocks(profile)) {
            return encodeHashes(profile, hashData);
        }
    }

    return 0;
}

bool Argon2::initializeSeeds(const Argon2Profile &profile, HashData &hashData) {
    if(m_prehash != NULL) {
        return (*m_prehash)(hashData.input, m_threads, (Argon2Profile*)&profile, m_userData);
    }
    else {
        uint8_t blockhash[ARGON2_PREHASH_SEED_LENGTH];

        for (int i = 0; i < m_threads; i++, (*(nonce(hashData)))++) {
            initialHash(profile, blockhash, (char *) hashData.input, hashData.inSize, xmrig::ARGON2_HASHLEN);

            memset(blockhash + ARGON2_PREHASH_DIGEST_LENGTH, 0,
                   ARGON2_PREHASH_SEED_LENGTH -
                   ARGON2_PREHASH_DIGEST_LENGTH);

            fillFirstBlocks(profile, blockhash, i);
        }

        return true;
    }
}

bool Argon2::fillBlocks(const Argon2Profile &profile) {
    m_outputMemory = (uint8_t *)(*m_filler) (m_threads, (Argon2Profile*)&profile, m_userData);
    return m_outputMemory != NULL;
}

int Argon2::encodeHashes(const Argon2Profile &profile, HashData &hashData) {
    if(m_posthash != NULL) {
        if((*m_posthash)(hashData.output, m_threads, (Argon2Profile*)&profile, m_userData)) {
            return m_threads;
        }
        return 0;
    }
    else {
        if (m_outputMemory != NULL) {
            uint32_t nonceInfo = *(nonce(hashData)) - m_threads;

            for (int i = 0; i < m_threads; i++, nonceInfo++) {
                blake2b_long((void *) (hashData.output + i * hashData.outSize), xmrig::ARGON2_HASHLEN,
                             (void *) (m_outputMemory + i * profile.memSize), ARGON2_BLOCK_SIZE);
                memcpy(hashData.output + i * hashData.outSize + xmrig::ARGON2_HASHLEN, &nonceInfo, 4);
            }
            return m_threads;
        }
        else
            return 0;
    }
}

void Argon2::initialHash(const Argon2Profile &profile, uint8_t *blockhash, const char *data, size_t dataSz,size_t outSz) {
    blake2b_state BlakeHash;
    uint32_t value;

    blake2b_init(&BlakeHash, ARGON2_PREHASH_DIGEST_LENGTH);

    value = profile.thrCost;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = outSz;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = profile.memCost;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = profile.tmCost;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = ARGON2_VERSION;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = ARGON2_TYPE_VALUE;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    value = (uint32_t)dataSz;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));
    blake2b_update(&BlakeHash, (const uint8_t *)data, dataSz);

    value = xmrig::ARGON2_SALTLEN;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));
    blake2b_update(&BlakeHash, (const uint8_t *)data, xmrig::ARGON2_SALTLEN);

    value = 0;
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));
    blake2b_update(&BlakeHash, (const uint8_t *)&value, sizeof(value));

    blake2b_final(&BlakeHash, blockhash, ARGON2_PREHASH_DIGEST_LENGTH);
}

void Argon2::fillFirstBlocks(const Argon2Profile &profile, uint8_t *blockhash, int thread) {
    block *blocks = (block *)(m_seedMemory + thread * profile.memSize);
    size_t lane_length = profile.memCost / profile.thrCost;

    for (uint32_t l = 0; l < profile.thrCost; ++l) {
        *((uint32_t*)(blockhash + ARGON2_PREHASH_DIGEST_LENGTH)) = 0;
        *((uint32_t*)(blockhash + ARGON2_PREHASH_DIGEST_LENGTH + 4)) = l;

        blake2b_long((void *)(blocks + l * lane_length), ARGON2_BLOCK_SIZE, blockhash,
                     ARGON2_PREHASH_SEED_LENGTH);

        *((uint32_t*)(blockhash + ARGON2_PREHASH_DIGEST_LENGTH)) = 1;

        blake2b_long((void *)(blocks + l * lane_length + 1), ARGON2_BLOCK_SIZE, blockhash,
                     ARGON2_PREHASH_SEED_LENGTH);
    }
}

void Argon2::setThreads(int threads) {
    m_threads = threads;
}
