#pragma once

#include "hash-ops.h"

typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

#ifdef __cplusplus

#include <string>

typedef std::string blobdata;

namespace crypto {
#pragma pack(push, 1)
  class hash {
    char data[HASH_SIZE];
  };
#pragma pack(pop)
}

#endif
