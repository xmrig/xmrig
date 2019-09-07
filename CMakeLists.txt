cmake_minimum_required(VERSION 2.8)
project(ninjarig)

option(WITH_LIBCPUID        "Use Libcpuid" ON)
option(WITH_HTTPD           "HTTP REST API" ON)
option(WITH_DEBUG_LOG       "Enable debug log output" OFF)
option(WITH_TLS             "Enable OpenSSL support" ON)
option(WITH_EMBEDDED_CONFIG "Enable internal embedded JSON config" OFF)
option(WITH_CUDA            "Enable CUDA support" ON)
option(WITH_OPENCL          "Enable OpenCL support" ON)
option(WITH_OPENCL_DOUBLE_THREADS  "Enable dual threads for OpenCL jobs" OFF)
option(WITH_CUDA_DOUBLE_THREADS  "Enable dual threads for CUDA jobs" ON)

include (CheckIncludeFile)
include (cmake/cpu.cmake)
include (cmake/TargetArch.cmake)

target_architecture (ARCH)
MESSAGE( STATUS "Target architecture is: " ${ARCH} )

SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "./")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set (CMAKE_MACOSX_RPATH 0)

set(HEADERS
    src/api/NetworkState.h
    src/App.h
    src/base/io/Json.h
    src/base/io/Watcher.h
    src/base/kernel/interfaces/IConfigListener.h
    src/base/kernel/interfaces/ISignalListener.h
    src/base/kernel/interfaces/IWatcherListener.h
    src/base/kernel/Entry.h
    src/base/kernel/Process.h
    src/base/kernel/Signals.h
    src/base/net/Pool.h
    src/base/net/Pools.h
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
    src/common/net/Storage.h
    src/common/net/strategies/FailoverStrategy.h
    src/common/net/strategies/SinglePoolStrategy.h
    src/common/net/SubmitResult.h
    src/common/Platform.h
    src/common/utils/c_str.h
    src/common/utils/mm_malloc.h
    src/common/xmrig.h
    src/core/ConfigLoader_platform.h
    src/core/ConfigLoader_default.h
    src/core/Controller.h
    src/interfaces/IJobResultListener.h
    src/interfaces/IWorker.h
    src/net/JobResult.h
    src/net/Network.h
    src/net/strategies/DonateStrategy.h
    src/net/strategies/Http.h
    src/Summary.h
    src/version.h
    src/core/HasherConfig.h
    src/workers/Handle.h
    src/workers/Hashrate.h
    src/workers/Worker.h
    src/workers/Workers.h
   )

set(HEADERS_CRYPTO
        src/crypto/Argon2_constants.h
        )

set(SOURCES
    src/api/NetworkState.cpp
    src/App.cpp
    src/base/io/Json.cpp
    src/base/io/Watcher.cpp
    src/base/kernel/Entry.cpp
    src/base/kernel/Process.cpp
    src/base/kernel/Signals.cpp
    src/base/net/Pool.cpp
    src/base/net/Pools.cpp
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
    src/common/net/strategies/FailoverStrategy.cpp
    src/common/net/strategies/SinglePoolStrategy.cpp
    src/common/net/SubmitResult.cpp
    src/common/Platform.cpp
    src/core/Config.cpp
    src/core/Controller.cpp
    src/net/Network.cpp
    src/net/strategies/DonateStrategy.cpp
    src/net/strategies/Http.cpp
    src/net/strategies/http_parser/http_parser.c
    src/Summary.cpp
    src/workers/Handle.cpp
    src/workers/Hashrate.cpp
    src/workers/Worker.cpp
    src/workers/Workers.cpp
    src/xmrig.cpp
   )

set(HEADERS_COMMON
        src/crypto/argon2_hasher/common/common.h
        src/crypto/argon2_hasher/common/DLLExport.h
        src/crypto/argon2_hasher/common/DLLImport.h
        src/crypto/argon2_hasher/crypt/base64.h
        src/crypto/argon2_hasher/crypt/hex.h
        src/crypto/argon2_hasher/crypt/random_generator.h
        src/crypto/argon2_hasher/crypt/sha512.h
        src/crypto/argon2_hasher/hash/argon2/blake2/blake2.h
        src/crypto/argon2_hasher/hash/argon2/blake2/blake2-config.h
        src/crypto/argon2_hasher/hash/argon2/blake2/blake2-impl.h
        src/crypto/argon2_hasher/hash/argon2/blake2/blake2b-load-sse2.h
        src/crypto/argon2_hasher/hash/argon2/blake2/blake2b-load-sse41.h
        src/crypto/argon2_hasher/hash/argon2/blake2/blake2b-round.h
        src/crypto/argon2_hasher/hash/argon2/Argon2.h
        src/crypto/argon2_hasher/hash/argon2/Defs.h
        src/crypto/argon2_hasher/hash/Hasher.h
        )

set(SOURCES_COMMON
        src/crypto/argon2_hasher/common/common.cpp
        src/crypto/argon2_hasher/crypt/base64.cpp
        src/crypto/argon2_hasher/crypt/hex.cpp
        src/crypto/argon2_hasher/crypt/random_generator.cpp
        src/crypto/argon2_hasher/crypt/sha512.cpp
        src/crypto/argon2_hasher/hash/argon2/blake2/blake2b.c
        src/crypto/argon2_hasher/hash/argon2/Argon2.cpp
        src/crypto/argon2_hasher/hash/argon2/argon2profile_4_1_256.c
        src/crypto/argon2_hasher/hash/argon2/argon2profile_3_1_512.c
        src/crypto/argon2_hasher/hash/Hasher.cpp
        src/core/HasherConfig.cpp)

set(SOURCE_CPU_HASHER src/crypto/argon2_hasher/hash/cpu/CpuHasher.cpp src/crypto/argon2_hasher/hash/cpu/CpuHasher.h)

set(SOURCE_OPENCL_HASHER src/crypto/argon2_hasher/hash/gpu/opencl/OpenCLHasher.cpp src/crypto/argon2_hasher/hash/gpu/opencl/OpenCLHasher.h
        src/crypto/argon2_hasher/hash/gpu/opencl/OpenCLKernel.cpp src/crypto/argon2_hasher/hash/gpu/opencl/OpenCLKernel.h)

set(SOURCE_CUDA_HASHER src/crypto/argon2_hasher/hash/gpu/cuda/CudaHasher.cpp src/crypto/argon2_hasher/hash/gpu/cuda/CudaHasher.h
        src/crypto/argon2_hasher/hash/gpu/cuda/cuda_kernel.cu)

set(ARGON2_FILL_BLOCKS_SRC
        src/crypto/argon2_hasher/hash/cpu/argon2_opt/implementation.c
        src/crypto/argon2_hasher/hash/cpu/argon2_opt/blamka-round-opt.h
        src/crypto/argon2_hasher/hash/cpu/argon2_opt/blamka-round-ref.h
        src/crypto/argon2_hasher/hash/argon2/Defs.h
        src/crypto/argon2_hasher/hash/argon2/blake2/blake2-impl.h)

if (WIN32)
    set(SOURCES_OS
        src/App_win.cpp
        src/base/io/Json_win.cpp
        src/common/Platform_win.cpp
        )

    add_definitions(/DWIN32)
    set(EXTRA_LIBS ws2_32 psapi iphlpapi userenv)
elseif (APPLE)
    set(SOURCES_OS
        src/App_unix.cpp
        src/base/io/Json_unix.cpp
        src/common/Platform_mac.cpp
    )
else()
    set(SOURCES_OS
        src/App_unix.cpp
        src/base/io/Json_unix.cpp
        src/common/Platform_unix.cpp
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

CHECK_INCLUDE_FILE (syslog.h HAVE_SYSLOG_H)
if (HAVE_SYSLOG_H)
    add_definitions(/DHAVE_SYSLOG_H)
    set(SOURCES_SYSLOG src/common/log/SysLog.h src/common/log/SysLog.cpp)
endif()

if (WITH_EMBEDDED_CONFIG)
    add_definitions(/DXMRIG_FEATURE_EMBEDDED_CONFIG)
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
include_directories(src/crypto/argon2_hasher/hash/cpu/cpu_features/include)

add_subdirectory(src/crypto/argon2_hasher/hash/cpu/cpu_features)
set_property(TARGET cpu_features PROPERTY POSITION_INDEPENDENT_CODE ON)

if (WITH_DEBUG_LOG)
    add_definitions(/DAPP_DEBUG)
endif()

if(WIN32)
    add_definitions(/DRAPIDJSON_HAS_STDSTRING=1)
    add_definitions(/DRAPIDJSON_HAS_CXX11_RVALUE_REFS=1)

    include_directories(src/win64/include)
    list(APPEND SOURCES_COMMON src/win64/src/dlfcn.c)
endif()

add_library(argon2_common SHARED ${HEADERS_COMMON} ${SOURCES_COMMON})
target_link_libraries(argon2_common ${CMAKE_DL_LIBS})

if(WIN32)
    target_link_libraries(argon2_common Psapi.lib)
endif()

add_executable(${CMAKE_PROJECT_NAME} ${HEADERS} ${SOURCES} ${SOURCES_OS} ${SOURCES_CPUID} ${HEADERS_CRYPTO} ${SOURCES_SYSLOG} ${HTTPD_SOURCES} ${TLS_SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${OPENSSL_LIBRARIES} ${UV_LIBRARIES} ${MHD_LIBRARY} ${EXTRA_LIBS} ${CPUID_LIB} argon2_common)

add_library(cpu_hasher MODULE ${SOURCE_CPU_HASHER})
set_target_properties(cpu_hasher
        PROPERTIES
        PREFIX ""
        SUFFIX ".hsh"
        LIBRARY_OUTPUT_DIRECTORY modules
        )
target_link_libraries(cpu_hasher argon2_common cpu_features)
add_dependencies(${CMAKE_PROJECT_NAME} cpu_hasher)

add_library(argon2_fill_blocks_REF MODULE ${ARGON2_FILL_BLOCKS_SRC})
set_target_properties(argon2_fill_blocks_REF
        PROPERTIES
        PREFIX ""
        SUFFIX ".opt"
        LIBRARY_OUTPUT_DIRECTORY modules
        )
target_compile_definitions(argon2_fill_blocks_REF PRIVATE BUILD_REF=1)
add_dependencies(cpu_hasher argon2_fill_blocks_REF)

if(ARCH STREQUAL "x86_64")
    add_library(argon2_fill_blocks_SSE2 MODULE ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_SSSE3 MODULE ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX MODULE ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX2 MODULE ${ARGON2_FILL_BLOCKS_SRC})
    add_library(argon2_fill_blocks_AVX512F MODULE ${ARGON2_FILL_BLOCKS_SRC})
    set_target_properties(argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F
            PROPERTIES
            PREFIX ""
            SUFFIX ".opt"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_compile_options(argon2_fill_blocks_SSE2 PRIVATE -msse2)
    target_compile_options(argon2_fill_blocks_SSSE3 PRIVATE -mssse3)
    target_compile_options(argon2_fill_blocks_AVX PRIVATE -mavx)
    target_compile_options(argon2_fill_blocks_AVX2 PRIVATE -mavx2)
    target_compile_options(argon2_fill_blocks_AVX512F PRIVATE -mavx512f)
    add_dependencies(cpu_hasher argon2_fill_blocks_SSE2 argon2_fill_blocks_SSSE3 argon2_fill_blocks_AVX argon2_fill_blocks_AVX2 argon2_fill_blocks_AVX512F)
endif()

if(ARCH STREQUAL "arm" OR ARCH STREQUAL "aarch64")
    add_library(argon2_fill_blocks_NEON MODULE ${ARGON2_FILL_BLOCKS_SRC})
    set_target_properties(argon2_fill_blocks_NEON
            PROPERTIES
            PREFIX ""
            SUFFIX ".opt"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_compile_options(argon2_common PRIVATE -D__NEON__)
    if(ARCH STREQUAL "arm")
        target_compile_options(argon2_fill_blocks_NEON PRIVATE -D__NEON__ -mfpu=neon -funsafe-math-optimizations)
    else()
        target_compile_options(argon2_fill_blocks_NEON PRIVATE -D__NEON__)
    endif(ARCH STREQUAL "arm")

    add_dependencies(cpu_hasher argon2_fill_blocks_NEON)
endif(ARCH STREQUAL "arm" OR ARCH STREQUAL "aarch64")

if(WITH_OPENCL)
    add_definitions(-DWITH_OPENCL)
    find_package(OpenCL REQUIRED)
    if(WITH_OPENCL_DOUBLE_THREADS)
        add_definitions(-DPARALLEL_OPENCL)
    endif()
    include_directories(${OpenCL_INCLUDE_DIR})
    add_library(opencl_hasher MODULE ${SOURCE_OPENCL_HASHER})
    set_target_properties(opencl_hasher
            PROPERTIES
            PREFIX ""
            SUFFIX ".hsh"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_link_libraries(opencl_hasher argon2_common ${OpenCL_LIBRARY})
    add_dependencies(${CMAKE_PROJECT_NAME} opencl_hasher)
endif()

if(WITH_CUDA)
    add_definitions(-DWITH_CUDA)
    find_package(CUDA REQUIRED)
    if(WITH_CUDA_DOUBLE_THREADS)
        add_definitions(-DPARALLEL_CUDA)
    endif()
    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -O3 -std=c++11 --ptxas-options=-v
            -gencode=arch=compute_75,code="sm_75,compute_75"
            -gencode=arch=compute_61,code="sm_61,compute_61"
            -gencode=arch=compute_52,code="sm_52,compute_52"
            -gencode=arch=compute_50,code="sm_50,compute_50"
            -gencode=arch=compute_35,code="sm_35,compute_35"
            -gencode=arch=compute_30,code="sm_30,compute_30"
    )
    cuda_add_library(cuda_hasher MODULE ${SOURCE_CUDA_HASHER})
    set_target_properties(cuda_hasher
            PROPERTIES
            PREFIX ""
            SUFFIX ".hsh"
            LIBRARY_OUTPUT_DIRECTORY modules
            )
    target_link_libraries(cuda_hasher argon2_common)
    add_dependencies(${CMAKE_PROJECT_NAME} cuda_hasher)
endif()



