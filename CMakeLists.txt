cmake_minimum_required(VERSION 2.8)
project(xmrig)

option(WITH_LIBCPUID  "Use Libcpuid" ON)
option(WITH_AEON      "CryptoNight-Lite support" ON)
option(WITH_SUMO      "CryptoNight-Heavy support" ON)
option(WITH_CN_PICO   "CryptoNight-Pico support" ON)
option(WITH_CN_GPU    "CryptoNight-GPU support" ON)
option(WITH_HTTPD     "HTTP REST API" ON)
option(WITH_DEBUG_LOG "Enable debug log output" OFF)
option(WITH_TLS       "Enable OpenSSL support" ON)
option(WITH_ASM       "Enable ASM PoW implementations" ON)
option(BUILD_STATIC   "Build static binary" OFF)
option(ARM_TARGET     "Force use specific ARM target 8 or 7" 0)

include (CheckIncludeFile)
include (cmake/cpu.cmake)


set(HEADERS
    src/api/NetworkState.h
    src/App.h
    src/base/io/Json.h
    src/base/io/Watcher.h
    src/base/kernel/interfaces/IConfigListener.h
    src/base/kernel/interfaces/IWatcherListener.h
    src/base/kernel/Entry.h
    src/base/kernel/Process.h
    src/base/tools/Arguments.h
    src/base/tools/Handle.h
    src/base/tools/String.h
    src/common/config/CommonConfig.h
    src/common/config/ConfigLoader.h
    src/common/config/ConfigWatcher.h
    src/common/Console.h
    src/common/cpu/Cpu.h
    src/common/crypto/Algorithm.h
    src/common/crypto/keccak.h
    src/common/interfaces/IClientListener.h
    src/common/interfaces/IConfig.h
    src/common/interfaces/IConfigCreator.h
    src/common/interfaces/IConsoleListener.h
    src/common/interfaces/IControllerListener.h
    src/common/interfaces/ICpuInfo.h
    src/common/interfaces/ILogBackend.h
    src/common/interfaces/IStrategy.h
    src/common/interfaces/IStrategyListener.h
    src/common/log/BasicLog.h
    src/common/log/ConsoleLog.h
    src/common/log/FileLog.h
    src/common/log/Log.h
    src/common/net/Client.h
    src/common/net/Id.h
    src/common/net/Job.h
    src/common/net/Pool.h
    src/common/net/Storage.h
    src/common/net/strategies/FailoverStrategy.h
    src/common/net/strategies/SinglePoolStrategy.h
    src/common/net/SubmitResult.h
    src/common/Platform.h
    src/common/utils/c_str.h
    src/common/utils/mm_malloc.h
    src/common/xmrig.h
    src/core/ConfigLoader_platform.h
    src/core/Controller.h
    src/interfaces/IJobResultListener.h
    src/interfaces/IThread.h
    src/interfaces/IWorker.h
    src/Mem.h
    src/net/JobResult.h
    src/net/Network.h
    src/net/strategies/DonateStrategy.h
    src/Summary.h
    src/version.h
    src/workers/CpuThread.h
    src/workers/Handle.h
    src/workers/Hashrate.h
    src/workers/MultiWorker.h
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
    src/api/NetworkState.cpp
    src/App.cpp
    src/base/io/Json.cpp
    src/base/io/Watcher.cpp
    src/base/kernel/Entry.cpp
    src/base/kernel/Process.cpp
    src/base/tools/Arguments.cpp
    src/base/tools/Handle.cpp
    src/base/tools/String.cpp
    src/common/config/CommonConfig.cpp
    src/common/config/ConfigLoader.cpp
    src/common/config/ConfigWatcher.cpp
    src/common/Console.cpp
    src/common/crypto/Algorithm.cpp
    src/common/crypto/keccak.cpp
    src/common/log/BasicLog.cpp
    src/common/log/ConsoleLog.cpp
    src/common/log/FileLog.cpp
    src/common/log/Log.cpp
    src/common/net/Client.cpp
    src/common/net/Job.cpp
    src/common/net/Pool.cpp
    src/common/net/strategies/FailoverStrategy.cpp
    src/common/net/strategies/SinglePoolStrategy.cpp
    src/common/net/SubmitResult.cpp
    src/common/Platform.cpp
    src/core/Config.cpp
    src/core/Controller.cpp
    src/Mem.cpp
    src/net/Network.cpp
    src/net/strategies/DonateStrategy.cpp
    src/Summary.cpp
    src/workers/CpuThread.cpp
    src/workers/Handle.cpp
    src/workers/Hashrate.cpp
    src/workers/MultiWorker.cpp
    src/workers/Worker.cpp
    src/workers/Workers.cpp
    src/xmrig.cpp
   )

set(SOURCES_CRYPTO
    src/crypto/c_groestl.c
    src/crypto/c_blake256.c
    src/crypto/c_jh.c
    src/crypto/c_skein.c
    src/crypto/CryptonightR_gen.cpp
   )

if (WIN32)
    set(SOURCES_OS
        res/app.rc
        src/App_win.cpp
        src/base/io/Json_win.cpp
        src/common/Platform_win.cpp
        src/Mem_win.cpp
        )

    add_definitions(/DWIN32)
    set(EXTRA_LIBS ws2_32 psapi iphlpapi userenv)
elseif (APPLE)
    set(SOURCES_OS
        src/App_unix.cpp
        src/base/io/Json_unix.cpp
        src/common/Platform_mac.cpp
        src/Mem_unix.cpp
        )
else()
    set(SOURCES_OS
        src/App_unix.cpp
        src/base/io/Json_unix.cpp
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

CHECK_INCLUDE_FILE (syslog.h HAVE_SYSLOG_H)
if (HAVE_SYSLOG_H)
    add_definitions(/DHAVE_SYSLOG_H)
    set(SOURCES_SYSLOG src/common/log/SysLog.h src/common/log/SysLog.cpp)
endif()

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

if (WITH_HTTPD)
    find_package(MHD)

    if (MHD_FOUND)
        include_directories(${MHD_INCLUDE_DIRS})
        set(HTTPD_SOURCES
            src/api/Api.h
            src/api/ApiRouter.h
            src/common/api/HttpBody.h
            src/common/api/Httpd.h
            src/common/api/HttpReply.h
            src/common/api/HttpRequest.h
            src/api/Api.cpp
            src/api/ApiRouter.cpp
            src/common/api/Httpd.cpp
            src/common/api/HttpRequest.cpp
            )
    else()
        message(FATAL_ERROR "microhttpd NOT found: use `-DWITH_HTTPD=OFF` to build without http deamon support")
    endif()
else()
    set(HTTPD_SOURCES "")
    set(MHD_LIBRARY "")
    add_definitions(/DXMRIG_NO_HTTPD)
    add_definitions(/DXMRIG_NO_API)
endif()

include_directories(src)
include_directories(src/3rdparty)
include_directories(${UV_INCLUDE_DIR})

if (BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS " -static")
endif()

if (WITH_DEBUG_LOG)
    add_definitions(/DAPP_DEBUG)
endif()

add_executable(${CMAKE_PROJECT_NAME} ${HEADERS} ${SOURCES} ${SOURCES_OS} ${SOURCES_CPUID} ${HEADERS_CRYPTO} ${SOURCES_CRYPTO} ${SOURCES_SYSLOG} ${HTTPD_SOURCES} ${TLS_SOURCES} ${XMRIG_ASM_SOURCES} ${CN_GPU_SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${XMRIG_ASM_LIBRARY} ${OPENSSL_LIBRARIES} ${UV_LIBRARIES} ${MHD_LIBRARY} ${EXTRA_LIBS} ${CPUID_LIB})
