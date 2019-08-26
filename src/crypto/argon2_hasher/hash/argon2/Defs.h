//
// Created by Haifa Bogdan Adnan on 06/08/2018.
//

#ifndef ARIOMINER_DEFS_H
#define ARIOMINER_DEFS_H

#define ARGON2_RAW_LENGTH               32
#define ARGON2_TYPE_VALUE               2
#define ARGON2_VERSION                  0x13

#define ARGON2_BLOCK_SIZE               1024
#define ARGON2_DWORDS_IN_BLOCK          ARGON2_BLOCK_SIZE / 4
#define ARGON2_QWORDS_IN_BLOCK          ARGON2_BLOCK_SIZE / 8
#define ARGON2_OWORDS_IN_BLOCK          ARGON2_BLOCK_SIZE / 16
#define ARGON2_HWORDS_IN_BLOCK          ARGON2_BLOCK_SIZE / 32
#define ARGON2_512BIT_WORDS_IN_BLOCK    ARGON2_BLOCK_SIZE / 64
#define ARGON2_PREHASH_DIGEST_LENGTH    64
#define ARGON2_PREHASH_SEED_LENGTH      72

#ifdef __cplusplus 
extern "C" {
#endif

typedef struct block_ { uint64_t v[ARGON2_QWORDS_IN_BLOCK]; } block;

typedef struct Argon2Profile_ {
    uint32_t memCost;
    uint32_t thrCost;
    uint32_t tmCost;
    size_t memSize;
    int32_t *blockRefs;
    size_t blockRefsSize;
    char profileName[15];
    int32_t *segments; // { start segment / current block, stop segment (excluding) / previous block, addressing type = 0 -> i, 1 -> d }
    uint32_t segSize;
    uint32_t segCount;
    uint32_t succesiveIdxs; // 0 - idx are precalculated, 1 - idx are successive
    int pwdLen; // in dwords
    int saltLen; // in dwords
} Argon2Profile;

extern DLLEXPORT Argon2Profile argon2profile_3_1_512;
extern DLLEXPORT Argon2Profile argon2profile_4_1_256;

#ifdef __cplusplus
}
#endif

#endif //ARIOMINER_DEFS_H
