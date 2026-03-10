/*
 * Copyright © 2010-2012, 2014 Université Bordeaux
 * Copyright © 2010 Cisco Systems, Inc.  All rights reserved.
 * Copyright © 2014 Inria.  All rights reserved.
 *
 * See COPYING in top-level directory.
 */

/* Internals for x86's cpuid.  */

#ifndef HWLOC_PRIVATE_CPUID_X86_H
#define HWLOC_PRIVATE_CPUID_X86_H

/* A macro for annotating memory as uninitialized when building with MSAN
 * (and otherwise having no effect). See below for why this is used with
 * our custom assembly.
 */
#ifdef __has_feature
#define HWLOC_HAS_FEATURE(name) __has_feature(name)
#else
#define HWLOC_HAS_FEATURE(name) 0
#endif
#if HWLOC_HAS_FEATURE(memory_sanitizer) || defined(MEMORY_SANITIZER)
#include <sanitizer/msan_interface.h>
#define HWLOC_ANNOTATE_MEMORY_IS_INITIALIZED(ptr, len) __msan_unpoison(ptr, len)
#else
#define HWLOC_ANNOTATE_MEMORY_IS_INITIALIZED(ptr, len)
#endif

#if (defined HWLOC_X86_32_ARCH) && (!defined HWLOC_HAVE_MSVC_CPUIDEX)
static __hwloc_inline int hwloc_have_x86_cpuid(void)
{
  int ret;
  unsigned tmp, tmp2;
  __asm__(
      "mov $0,%0\n\t"   /* Not supported a priori */

      "pushfl   \n\t"   /* Save flags */

      "pushfl   \n\t"                                           \
      "pop %1   \n\t"   /* Get flags */                         \

#define TRY_TOGGLE                                              \
      "xor $0x00200000,%1\n\t"        /* Try to toggle ID */    \
      "mov %1,%2\n\t"   /* Save expected value */               \
      "push %1  \n\t"                                           \
      "popfl    \n\t"   /* Try to toggle */                     \
      "pushfl   \n\t"                                           \
      "pop %1   \n\t"                                           \
      "cmp %1,%2\n\t"   /* Compare with expected value */       \
      "jnz 0f\n\t"   /* Unexpected, failure */               \

      TRY_TOGGLE        /* Try to set/clear */
      TRY_TOGGLE        /* Try to clear/set */

      "mov $1,%0\n\t"   /* Passed the test! */

      "0: \n\t"
      "popfl    \n\t"   /* Restore flags */

      : "=r" (ret), "=&r" (tmp), "=&r" (tmp2));
  return ret;
}
#endif /* !defined HWLOC_X86_32_ARCH && !defined HWLOC_HAVE_MSVC_CPUIDEX*/
#if (defined HWLOC_X86_64_ARCH) || (defined HWLOC_HAVE_MSVC_CPUIDEX)
static __hwloc_inline int hwloc_have_x86_cpuid(void) { return 1; }
#endif /* HWLOC_X86_64_ARCH */

static __hwloc_inline void hwloc_x86_cpuid(unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx)
{
#ifdef HWLOC_HAVE_MSVC_CPUIDEX
  int regs[4];
  __cpuidex(regs, *eax, *ecx);
  *eax = regs[0];
  *ebx = regs[1];
  *ecx = regs[2];
  *edx = regs[3];
#else /* HWLOC_HAVE_MSVC_CPUIDEX */
  /* Note: gcc might want to use bx or the stack for %1 addressing, so we can't
   * use them :/ */
#ifdef HWLOC_X86_64_ARCH
  hwloc_uint64_t sav_rbx;
  __asm__(
  "mov %%rbx,%2\n\t"
  "cpuid\n\t"
  "xchg %2,%%rbx\n\t"
  "movl %k2,%1\n\t"
  : "+a" (*eax), "=m" (*ebx), "=&r"(sav_rbx),
    "+c" (*ecx), "=&d" (*edx));
  /* MSAN does not recognize the effect of the above assembly on the memory operand
   * (`"=m"(*ebx)`). This may get improved in MSAN at some point in the future, e.g.
   * see https://github.com/llvm/llvm-project/pull/77393. */
  HWLOC_ANNOTATE_MEMORY_IS_INITIALIZED(ebx, sizeof *ebx);
#elif defined(HWLOC_X86_32_ARCH)
  __asm__(
  "mov %%ebx,%1\n\t"
  "cpuid\n\t"
  "xchg %%ebx,%1\n\t"
  : "+a" (*eax), "=&SD" (*ebx), "+c" (*ecx), "=&d" (*edx));
  /* See above. */
  HWLOC_ANNOTATE_MEMORY_IS_INITIALIZED(ebx, sizeof *ebx);
#else
#error unknown architecture
#endif
#endif /* HWLOC_HAVE_MSVC_CPUIDEX */
}

#endif /* HWLOC_PRIVATE_X86_CPUID_H */
