//
// Created by Haifa Bogdan Adnan on 05/08/2018.
//

#ifndef ARIOMINER_ARGON2_H
#define ARIOMINER_ARGON2_H

#include "Defs.h"
#include "crypto/argon2_hasher/hash/Hasher.h"

typedef bool (*argon2BlocksPrehash)(void *, int, Argon2Profile *, void *); // data_memory
typedef void *(*argon2BlocksFillerPtr)(int, Argon2Profile *, void *);
typedef bool (*argon2BlocksPosthash)(void *, int, Argon2Profile *, void *); // raw_hash_mem

struct HashData {
    uint8_t *input;
    uint8_t *output;
    size_t inSize;
    size_t outSize;
};

class DLLEXPORT Argon2 {
public:
    Argon2(argon2BlocksPrehash prehash, argon2BlocksFillerPtr filler, argon2BlocksPosthash posthash, void *memory, void *userData);

    int generateHashes(const Argon2Profile &profile, HashData &hashData);

    bool initializeSeeds(const Argon2Profile &profile, HashData &hashData);
    bool fillBlocks(const Argon2Profile &profile);
	int encodeHashes(const Argon2Profile &profile, HashData &hashData);

    void setThreads(int threads);

private:
    void initialHash(const Argon2Profile &profile, uint8_t *blockhash, const char *data, size_t dataSz, size_t outSz);
    void fillFirstBlocks(const Argon2Profile &profile, uint8_t *blockhash, int thread);

    inline uint32_t *nonce(HashData &hashData)
    {
        return reinterpret_cast<uint32_t*>(hashData.input + 39);
    }

	argon2BlocksPrehash m_prehash;
	argon2BlocksFillerPtr m_filler;
	argon2BlocksPosthash m_posthash;

    int m_threads;

    uint8_t *m_seedMemory;
	uint8_t *m_outputMemory;

    void *m_userData;
};


#endif //ARIOMINER_ARGON2_H
