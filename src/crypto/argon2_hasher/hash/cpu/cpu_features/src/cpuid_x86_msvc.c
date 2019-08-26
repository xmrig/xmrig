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

#if defined(CPU_FEATURES_ARCH_X86) && defined(CPU_FEATURES_COMPILER_MSC)
#include <immintrin.h>
#include <intrin.h>  // For __cpuidex()

Leaf CpuId(uint32_t leaf_id) {
  Leaf leaf;
  int data[4];
  __cpuid(data, leaf_id);
  leaf.eax = data[0];
  leaf.ebx = data[1];
  leaf.ecx = data[2];
  leaf.edx = data[3];
  return leaf;
}

uint32_t GetXCR0Eax(void) { return _xgetbv(0); }

#endif  // defined(CPU_FEATURES_ARCH_X86) && defined(CPU_FEATURES_COMPILER_MSC)
