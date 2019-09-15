#ifndef ARGON2_CPU_FLAGS_H
#define ARGON2_CPU_FLAGS_H

void cpu_flags_get(void);

int cpu_flags_have_sse2(void);
int cpu_flags_have_ssse3(void);
int cpu_flags_have_xop(void);
int cpu_flags_have_avx2(void);
int cpu_flags_have_avx512f(void);

#endif // ARGON2_CPU_FLAGS_H
