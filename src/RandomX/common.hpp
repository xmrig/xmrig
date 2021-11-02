#pragma once

/*
Copyright (c) 2019 SChernykh

This file is part of RandomX CUDA.

RandomX CUDA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX CUDA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX CUDA.  If not, see<http://www.gnu.org/licenses/>.
*/


#include <cstdint>


#define RANDOMX_DATASET_ITEM_SIZE  64
#define RANDOMX_DATASET_EXTRA_SIZE 33554368
#define RANDOMX_JUMP_BITS          8
#define RANDOMX_JUMP_OFFSET        8


namespace randomx {
    constexpr int mantissaSize = 52;
    constexpr int exponentSize = 11;
    constexpr uint64_t mantissaMask = (1ULL << mantissaSize) - 1;
    constexpr uint64_t exponentMask = (1ULL << exponentSize) - 1;
    constexpr int exponentBias = 1023;
    constexpr int dynamicExponentBits = 4;
    constexpr int staticExponentBits = 4;
    constexpr uint64_t constExponentBits = 0x300;
    constexpr uint64_t dynamicMantissaMask = (1ULL << (mantissaSize + dynamicExponentBits)) - 1;

    constexpr int RegistersCount = 8;
    constexpr int RegisterCountFlt = RegistersCount / 2;
    constexpr int RegisterNeedsDisplacement = 5; //x86 r13 register

    constexpr int CacheLineSize = RANDOMX_DATASET_ITEM_SIZE;
    constexpr uint32_t DatasetExtraItems = RANDOMX_DATASET_EXTRA_SIZE / RANDOMX_DATASET_ITEM_SIZE;

    constexpr uint32_t ConditionMask = ((1 << RANDOMX_JUMP_BITS) - 1);
    constexpr int ConditionOffset = RANDOMX_JUMP_OFFSET;
    constexpr int StoreL3Condition = 14;
}

#include "blake2b_cuda.hpp"
#include "aes_cuda.hpp"
#include "randomx_cuda.hpp"
#include "hash.hpp"
