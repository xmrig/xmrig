cmake_minimum_required(VERSION 2.8)
project(xmrig)

option(WITH_LIBCPUID        "Use Libcpuid" ON)
option(WITH_AEON            "CryptoNight-Lite support" ON)
option(WITH_SUMO            "CryptoNight-Heavy support" ON)
option(WITH_CN_PICO         "CryptoNight-Pico support" ON)
option(WITH_CN_GPU          "CryptoNight-GPU support" ON)
option(WITH_HTTP            "HTTP protocol support (client/server)" ON)
option(WITH_DEBUG_LOG       "Enable debug log output" OFF)
option(WITH_TLS             "Enable OpenSSL support" ON)
option(WITH_ASM             "Enable ASM PoW implementations" ON)
option(BUILD_STATIC         "Build static binary" OFF)
option(ARM_TARGET           "Force use specific ARM target 8 or 7" 0)
option(WITH_EMBEDDED_CONFIG "Enable internal embedded JSON config" OFF)

include (CheckIncludeFile)
include (cmake/cpu.cmake)
include (src/base/base.cmake)


set(HEADERS
    "${HEADERS_BASE}"
    "${HEADERS_BASE_HTTP}"
    src/api/interfaces/IApiListener.h
    src/App.h
    src/common/cpu/Cpu.h
    src/common/crypto/Algorithm.h
    src/common/crypto/keccak.h
    src/common/interfaces/ICpuInfo.h
    src/common/Platform.h
    src/common/utils/mm_malloc.h
    src/common/xmrig.h
    src/core/config/Config_default.h
    src/core/config/Config_platform.h
    src/core/config/Config.h
    src/core/config/ConfigTransform.h
    src/core/config/usage.h
    src/core/Controller.h
    src/interfaces/IJobResultListener.h
    src/interfaces/IThread.h
    src/interfaces/IWorker.h
    src/Mem.h
    src/net/JobResult.h
    src/net/Network.h
    src/net/NetworkState.h
    src/net/strategies/DonateStrategy.h
    src/Summary.h
    src/version.h
    src/workers/CpuThread.h
    src/workers/Hashrate.h
    src/workers/MultiWorker.h
    src/workers/ThreadHandle.h
    src/workers/Worker.h
    src/workers/Workers.h
   )

set(HEADERS_CRYPTO
    src/crypto/c_blake256.h
    src/crypto/c_groestl.h
    src/crypto/c_jh.h
    src/crypto/c_skein.h
    src/crypto/CryptoNight.h
    src/crypto/CryptoNight_constants.h
    src/crypto/CryptoNight_monero.h
    src/crypto/CryptoNight_test.h
    src/crypto/groestl_tables.h
    src/crypto/hash.h
    src/crypto/skein_port.h
    src/crypto/soft_aes.h
    src/crypto/asm/CryptonightR_template.h
   )

if (XMRIG_ARM)
    set(HEADERS_CRYPTO "${HEADERS_CRYPTO}" src/crypto/CryptoNight_arm.h)
else()
    set(HEADERS_CRYPTO "${HEADERS_CRYPTO}" src/crypto/CryptoNight_x86.h)
endif()

set(SOURCES
    "${SOURCES_BASE}"
    "${SOURCES_BASE_HTTP}"
    src/App.cpp
    src/common/crypto/Algorithm.cpp
    src/common/crypto/keccak.cpp
    src/common/Platform.cpp
    src/core/config/Config.cpp
    src/core/config/ConfigTransform.cpp
    src/core/Controller.cpp
    src/Mem.cpp
    src/net/Network.cpp
    src/net/NetworkState.cpp
    src/net/strategies/DonateStrategy.cpp
    src/Summary.cpp
    src/workers/CpuThread.cpp
    src/workers/Hashrate.cpp
    src/workers/MultiWorker.cpp
    src/workers/ThreadHandle.cpp
    src/workers/Worker.cpp
    src/workers/Workers.cpp
    src/xmrig.cpp
   )

set(SOURCES_CRYPTO
    src/crypto/c_groestl.c
    src/crypto/c_blake256.c
    src/crypto/c_jh.c
    src/crypto/c_skein.c
   )

if (WIN32)
    set(SOURCES_OS
        "${SOURCES_OS}"
        res/app.rc
        src/App_win.cpp
        src/common/Platform_win.cpp
        src/Mem_win.cpp
        )

    add_definitions(/DWIN32)
    set(EXTRA_LIBS ws2_32 psapi iphlpapi userenv)
elseif (APPLE)
    set(SOURCES_OS
        "${SOURCES_OS}"
        src/App_unix.cpp
        src/common/Platform_mac.cpp
        src/Mem_unix.cpp
        )
else()
    set(SOURCES_OS
        "${SOURCES_OS}"
        src/App_unix.cpp
        src/common/Platform_unix.cpp
        src/Mem_unix.cpp
        )

    if (CMAKE_SYSTEM_NAME STREQUAL FreeBSD)
        set(EXTRA_LIBS kvm pthread)
    else()
        set(EXTRA_LIBS pthread rt dl)
    endif()
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    EXECUTE_PROCESS(COMMAND uname -o COMMAND tr -d '\n' OUTPUT_VARIABLE OPERATING_SYSTEM)
    if (OPERATING_SYSTEM MATCHES "Android")
        set(EXTRA_LIBS ${EXTRA_LIBS} log)
    endif()
endif()

add_definitions(/D__STDC_FORMAT_MACROS)
add_definitions(/DUNICODE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(UV REQUIRED)
find_package(RandomX REQUIRED)

include(cmake/flags.cmake)

if (WITH_LIBCPUID)
    add_subdirectory(src/3rdparty/libcpuid)

    include_directories(src/3rdparty/libcpuid)
    set(CPUID_LIB cpuid)
    set(SOURCES_CPUID src/core/cpu/AdvancedCpuInfo.h src/core/cpu/AdvancedCpuInfo.cpp src/core/cpu/Cpu.cpp)
else()
    add_definitions(/DXMRIG_NO_LIBCPUID)
    set(SOURCES_CPUID src/common/cpu/BasicCpuInfo.h src/common/cpu/Cpu.cpp)

    if (XMRIG_ARM)
        set(SOURCES_CPUID ${SOURCES_CPUID} src/common/cpu/BasicCpuInfo_arm.cpp)
    else()
        set(SOURCES_CPUID ${SOURCES_CPUID} src/common/cpu/BasicCpuInfo.cpp)
    endif()
endif()

include(cmake/OpenSSL.cmake)
include(cmake/asm.cmake)
include(cmake/cn-gpu.cmake)

if (NOT WITH_AEON)
    add_definitions(/DXMRIG_NO_AEON)
endif()

if (NOT WITH_SUMO)
    add_definitions(/DXMRIG_NO_SUMO)
endif()

if (NOT WITH_IPBC)
    add_definitions(/DXMRIG_NO_IPBC)
endif()

if (NOT WITH_CN_PICO)
    add_definitions(/DXMRIG_NO_CN_PICO)
endif()

if (WITH_EMBEDDED_CONFIG)
    add_definitions(/DXMRIG_FEATURE_EMBEDDED_CONFIG)
endif()

if (WITH_HTTP)
    set(HTTP_SOURCES
        src/api/Api.cpp
        src/api/Api.h
        src/api/Httpd.cpp
        src/api/Httpd.h
        src/api/interfaces/IApiRequest.h
        src/api/requests/ApiRequest.cpp
        src/api/requests/ApiRequest.h
        src/api/requests/HttpApiRequest.cpp
        src/api/requests/HttpApiRequest.h
        src/api/v1/ApiRouter.cpp
        src/api/v1/ApiRouter.h
        )
else()
    set(HTTP_SOURCES "")
endif()

include_directories(src)
include_directories(src/3rdparty)
include_directories(${UV_INCLUDE_DIR})
include_directories(${RANDOMX_INCLUDE_DIR})

if (BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS " -static")
endif()

if (WITH_DEBUG_LOG)
    add_definitions(/DAPP_DEBUG)
endif()

add_executable(${CMAKE_PROJECT_NAME} ${HEADERS} ${SOURCES} ${SOURCES_OS} ${SOURCES_CPUID} ${HEADERS_CRYPTO} ${SOURCES_CRYPTO} ${SOURCES_SYSLOG} ${HTTP_SOURCES} ${TLS_SOURCES} ${XMRIG_ASM_SOURCES} ${CN_GPU_SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${XMRIG_ASM_LIBRARY} ${OPENSSL_LIBRARIES} ${UV_LIBRARIES} ${RANDOMX_LIBRARIES} ${EXTRA_LIBS} ${CPUID_LIB})
