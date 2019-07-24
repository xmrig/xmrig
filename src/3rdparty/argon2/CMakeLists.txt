cmake_minimum_required(VERSION 2.6)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "-- Argon2: Found ccache package... Activating...")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

project(Argon2 C)
set(ARGON2_VERSION 1.0)
set(CMAKE_C_STANDARD 90)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(CheckCSourceCompiles)
find_package(Threads REQUIRED)

add_library(argon2-interface INTERFACE)
target_include_directories(argon2-interface INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(argon2-internal INTERFACE)
target_include_directories(argon2-internal INTERFACE lib lib/blake2)
target_link_libraries(argon2-internal INTERFACE argon2-interface)

set(ARGON2_SRC
  lib/argon2.c
  lib/core.c
  lib/encoding.c
  lib/genkat.c
  lib/impl-select.c
  lib/thread.c
  lib/blake2/blake2.c
)

message("-- Argon2: Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message("-- Argon2: Build Type: ${ARCH}")

if((CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64") AND NOT "${ARCH}" STREQUAL "default")
  include(CheckCXXSourceRuns)

  # Check for AVX2
  check_cxx_source_runs("
    #include <immintrin.h>
    int main()
    {
      __m256i a, b, c;
      const int src[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
      int dst[8];
      a =  _mm256_loadu_si256( (__m256i*)src );
      b =  _mm256_loadu_si256( (__m256i*)src );
      c = _mm256_add_epi32( a, b );
      _mm256_storeu_si256( (__m256i*)dst, c );
      for( int i = 0; i < 8; i++ ){
        if( ( src[i] + src[i] ) != dst[i] ){
          return -1;
        }
      }
      return 0;
    }"
  HAVE_AVX2_EXTENSIONS)

  if(HAVE_AVX2_EXTENSIONS)
    message("-- Argon2: AVX2 Extensions - Enabled")
    add_definitions(-DHAVE_AVX2)
    if(MSVC)
      add_definitions(/arch:AVX2)
    endif()
  else()
    message("-- Argon2: AVX2 Extensions - Disabled")
  endif()

  # Check for AVX512
  check_cxx_source_runs("
    #include <immintrin.h>
    int main()
    {
      __m512i a, b, c;
      const int src[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
      int dst[8];
      a =  _mm512_loadu_si512( (__m512i*)src );
      b =  _mm512_loadu_si512( (__m512i*)src );
      c = _mm512_add_epi32( a, b );
      _mm512_storeu_si512( (__m512i*)dst, c );
      for( int i = 0; i < 8; i++ ){
        if( ( src[i] + src[i] ) != dst[i] ){
          return -1;
        }
      }
      return 0;
    }"
  HAVE_AVX512F_EXTENSIONS)

  if(HAVE_AVX512F_EXTENSIONS)
    message("-- Argon2: AVX512 Extensions - Enabled")
    add_definitions(-DHAVE_AVX512F)
  else()
    message("-- Argon2: AVX512 Extensions - Disabled")
  endif()

  # Check for SSE2
  check_cxx_source_runs("
    #include <immintrin.h>
    int main()
    {
      __m128d a, b;
      double vals[2] = {0};
      a = _mm_loadu_pd(vals);
      b = _mm_add_pd(a,a);
      _mm_storeu_pd(vals,b);
      return 0;
    }"
  HAVE_SSE2_EXTENSIONS)

  if(HAVE_SSE2_EXTENSIONS)
    message("-- Argon2: SSE2 Extensions - Enabled")
    add_definitions(-DHAVE_SSE2)
    if(MSVC)
      add_definitions(/arch:SSE2)
    endif()
  else()
    message("-- Argon2: SSE2 Extensions - Disabled")
  endif()

  # Check for SSE3
  check_cxx_source_runs("
    #include <immintrin.h>
    int main()
    {
      __m128d a, b;
      double vals[2] = {0};
      a = _mm_loadu_pd(vals);
      b = _mm_hadd_pd(a,a);
      _mm_storeu_pd(vals, b);
      return 0;
    }"
  HAVE_SSE3_EXTENSIONS)

  if(HAVE_SSE3_EXTENSIONS)
    message("-- Argon2: SSE3 Extensions - Enabled")
    add_definitions(-DHAVE_SSE3)
    if(MSVC)
      add_definitions(/arch:SSE3)
    endif()
  else()
    message("-- Argon2: SSE3 Extensions - Disabled")
  endif()

  # Check for XOP
  check_cxx_source_runs("
    #include <immintrin.h>
    int main()
    {
      __m128i a, b, c;
      const int src[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
      a = _mm_load_si128( (__m128i*)src );
      b = _mm_load_si128( (__m128i*)src );
      c = _mm_roti_epi64(*a, b);
      return 0;
    }"
  HAVE_XOP_EXTENSIONS)

  if(HAVE_XOP_EXTENSIONS)
    message("-- Argon2: XOP Extensions - Enabled")
    add_definitions(-DHAVE_XOP)
  else()
    message("-- Argon2: XOP Extensions - Disabled")
  endif()

  list(APPEND ARGON2_SRC
    arch/x86_64/lib/argon2-sse2.c
    arch/x86_64/lib/argon2-ssse3.c
    arch/x86_64/lib/argon2-xop.c
    arch/x86_64/lib/argon2-avx2.c
    arch/x86_64/lib/argon2-avx512f.c
    arch/x86_64/lib/argon2-arch.c
  )
else()
  list(APPEND ARGON2_SRC
    arch/generic/lib/argon2-arch.c
  )
endif()

add_library(argon2 STATIC ${ARGON2_SRC})

target_compile_definitions(argon2
  PUBLIC "A2_VISCTL"
)

target_link_libraries(argon2
  PUBLIC argon2-interface ${CMAKE_THREAD_LIBS_INIT}
  PRIVATE argon2-internal
)

set_property(TARGET argon2 PROPERTY C_STANDARD 90)
set_property(TARGET argon2 PROPERTY VERSION ${Upstream_VERSION})
set_property(TARGET argon2 PROPERTY SOVERSION 1)
set_property(TARGET argon2 PROPERTY INTERFACE_ARGON2_MAJOR_VERSION 1)
set_property(TARGET argon2 APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING ARGON2_MAJOR_VERSION
)

