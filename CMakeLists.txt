cmake_minimum_required(VERSION 3.0)
project(xmrig C)

option(WITH_LIBCPUID "Use Libcpuid" ON)
option(WITH_AEON     "CryptoNight-Lite support" ON)

set(HEADERS
    algo/cryptonight/cryptonight.h
    algo/cryptonight/cryptonight_aesni.h
    algo/cryptonight/cryptonight_monero.h
    algo/cryptonight/cryptonight_softaes.h
    algo/cryptonight/cryptonight_test.h
    compat.h
    cpu.h
    donate.h
    elist.h
    options.h
    persistent_memory.h
    stats.h
    stratum.h
    util.h
    version.h
    xmrig.h
   )

set(HEADERS_CRYPTO
    crypto/c_groestl.h
    crypto/c_blake256.h
    crypto/c_jh.h
    crypto/c_skein.h
   )

set(HEADERS_COMPAT
    compat/winansi.h
   )

set(HEADERS_UTILS
    utils/applog.h
    utils/threads.h
    utils/summary.h
   )

set(SOURCES
    xmrig.c
    algo/cryptonight/cryptonight.c
    algo/cryptonight/cryptonight_av1.c
    algo/cryptonight/cryptonight_av2.c
    algo/cryptonight/cryptonight_av3.c
    algo/cryptonight/cryptonight_av4.c
    util.c
    options.c
    stratum.c
    stats.c
    memory.c
   )

set(SOURCES_CRYPTO
    crypto/c_keccak.c
    crypto/c_groestl.c
    crypto/c_blake256.c
    crypto/c_jh.c
    crypto/c_skein.c
    crypto/soft_aes.c
   )

set(SOURCES_UTILS
    utils/applog.c
    utils/summary.c
   )

if (WIN32)
    set(SOURCES_OS win/cpu_win.c win/memory_win.c win/xmrig_win.c win/app.rc compat/winansi.c)
    set(EXTRA_LIBS ws2_32 crypt32)
    add_definitions(/D_WIN32_WINNT=0x600)
elseif (APPLE)
    set(SOURCES_OS mac/cpu_mac.c mac/memory_mac.c mac/xmrig_mac.c)
else()
    set(SOURCES_OS unix/cpu_unix.c unix/memory_unix.c unix/xmrig_unix.c)
    set(EXTRA_LIBS pthread rt m)
endif()

include_directories(.)
add_definitions(/DUSE_NATIVE_THREADS)
add_definitions(/D_GNU_SOURCE)
add_definitions(/DUNICODE)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes -Wno-pointer-to-int-cast")

if (CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -s -funroll-loops -fvariable-expansion-in-unroller -fmerge-all-constants")
else()
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -s -funroll-loops -fvariable-expansion-in-unroller -ftree-loop-if-convert-stores -fmerge-all-constants -fbranch-target-load-optimize2")
endif()

#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -gdwarf-2")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fprofile-generate")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fprofile-use -fprofile-correction")

if (WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
endif()

include_directories(compat/jansson)
add_subdirectory(compat/jansson)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})
add_definitions(/DCURL_STATICLIB)
link_directories(${CURL_LIBRARIES})

if (WITH_LIBCPUID)
    add_subdirectory(compat/libcpuid)

    include_directories(compat/libcpuid)
    set(CPUID_LIB cpuid)
    set(SOURCES_CPUID cpu.c)
else()
    add_definitions(/DXMRIG_NO_LIBCPUID)
    set(SOURCES_CPUID cpu_stub.c)
endif()

if (WITH_AEON)
    set(SOURCES_AEON
    algo/cryptonight-lite/cryptonight_lite_av1.c
    algo/cryptonight-lite/cryptonight_lite_av2.c
    algo/cryptonight-lite/cryptonight_lite_av3.c
    algo/cryptonight-lite/cryptonight_lite_av4.c
    algo/cryptonight-lite/cryptonight_lite_aesni.h
    algo/cryptonight-lite/cryptonight_lite_softaes.h
    )
else()
    add_definitions(/DXMRIG_NO_AEON)
endif()

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    add_executable(xmrig ${HEADERS} ${HEADERS_CRYPTO} ${SOURCES} ${SOURCES_CRYPTO} ${HEADERS_UTILS} ${SOURCES_UTILS} ${HEADERS_COMPAT} ${SOURCES_COMPAT} ${SOURCES_OS} ${SOURCES_CPUID} ${SOURCES_AEON})
    target_link_libraries(xmrig jansson ${CURL_LIBRARY} ${CPUID_LIB} ${EXTRA_LIBS})
else()
    add_executable(xmrig32 ${HEADERS} ${HEADERS_CRYPTO} ${SOURCES} ${SOURCES_CRYPTO} ${HEADERS_UTILS} ${SOURCES_UTILS} ${HEADERS_COMPAT} ${SOURCES_COMPAT} ${SOURCES_OS} ${SOURCES_CPUID} ${SOURCES_AEON})
    target_link_libraries(xmrig32 jansson ${CURL_LIBRARY} ${CPUID_LIB} ${EXTRA_LIBS})
endif()

