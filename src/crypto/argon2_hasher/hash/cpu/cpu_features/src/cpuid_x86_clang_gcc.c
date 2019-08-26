// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "internal/cpuid_x86.h"

#if defined(CPU_FEATURES_ARCH_X86)
#if defined(CPU_FEATURES_COMPILER_CLANG) || defined(CPU_FEATURES_COMPILER_GCC)

#include <cpuid.h>

Leaf CpuId(uint32_t leaf_id) {
  Leaf leaf;
  __cpuid_count(leaf_id, 0, leaf.eax, leaf.ebx, leaf.ecx, leaf.edx);
  return leaf;
}

uint32_t GetXCR0Eax(void) {
  uint32_t eax, edx;
  __asm("XGETBV" : "=a"(eax), "=d"(edx) : "c"(0));
  return eax;
}

#endif  // defined(CPU_FEATURES_COMPILER_CLANG) ||
        // defined(CPU_FEATURES_COMPILER_GCC)
#endif  // defined(CPU_FEATURES_ARCH_X86)
