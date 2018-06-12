cmake_minimum_required(VERSION 2.8)
project(xmrig)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

option(WITH_LIBCPUID  "Use Libcpuid" ON)
option(WITH_HTTPD     "HTTP REST API" OFF)
option(WITH_CC_CLIENT "CC Client" ON)
option(WITH_CC_SERVER "CC Server" ON)
option(WITH_TLS       "TLS support" ON)
option(BUILD_STATIC   "Build static binary" OFF)
set(MINER_EXECUTABLE_NAME "xmrigMiner" CACHE STRING "Miner executable file name")
set(DAEMON_EXECUTABLE_NAME "xmrigDaemon" CACHE STRING "Daemon executable file name")

set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_USE_STATIC_LIBS ON)

include (CheckIncludeFile)
include (cmake/cpu.cmake)

set(SOURCES
    src/api/NetworkState.cpp
    src/App.cpp
    src/net/BoostTcpConnection.cpp
    src/net/Client.cpp
    src/net/Connection.cpp
    src/net/Job.cpp
    src/net/Network.cpp
    src/net/strategies/DonateStrategy.cpp
    src/net/strategies/FailoverStrategy.cpp
    src/net/strategies/SinglePoolStrategy.cpp
    src/net/SubmitResult.cpp
    src/Summary.cpp
    src/workers/MultiWorker.cpp
    src/workers/Handle.cpp
    src/workers/Hashrate.cpp
    src/workers/Worker.cpp
    src/workers/Workers.cpp
    src/xmrig.cpp
   )

set(SOURCES_CRYPTO
    src/crypto/c_keccak.c
    src/crypto/c_groestl.c
    src/crypto/c_blake256.c
    src/crypto/c_jh.c
    src/crypto/c_skein.c
    src/crypto/CryptoNight.cpp
   )

set(SOURCES_COMMON
    src/Console.cpp
    src/Mem.cpp
    src/net/Url.cpp
    src/Options.cpp
    src/log/ConsoleLog.cpp
    src/log/FileLog.cpp
    src/log/RemoteLog.cpp
    src/log/Log.cpp
    src/Platform.cpp
    src/Cpu.cpp

   )

if (WIN32)
    set(SOURCES_OS
        res/app.rc
        src/api/Api.cpp
        src/api/ApiState.cpp
        src/App_win.cpp
        src/Cpu_win.cpp
        src/Mem_win.cpp
        src/Platform_win.cpp
        )

    add_definitions(/DWIN32)
    set(EXTRA_LIBS ws2_32 psapi iphlpapi userenv crypt32)
elseif (APPLE)
    set(SOURCES_OS
        src/App_unix.cpp
        src/Cpu_mac.cpp
        src/Mem_unix.cpp
        src/Platform_mac.cpp
        )
else()
    set(SOURCES_OS
        src/api/Api.cpp
        src/api/ApiState.cpp
        src/App_unix.cpp
        src/Cpu_unix.cpp
        src/Mem_unix.cpp
        src/Platform_unix.cpp
        )

    set(EXTRA_LIBS pthread rt)

    if (CMAKE_SYSTEM_NAME STREQUAL FreeBSD)
    	set(EXTRA_LIBS ${EXTRA_LIBS} kvm)
    else()
    	set(EXTRA_LIBS ${EXTRA_LIBS} dl)
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
add_definitions(/DMINER_EXECUTABLE_NAME=${MINER_EXECUTABLE_NAME})
#add_definitions(/DAPP_DEBUG)

add_compile_options(-fexceptions)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(UV REQUIRED)

if (WIN32)
    add_definitions(-DBOOST_ALL_NO_LIB)
endif()

find_package(Boost 1.63.0 COMPONENTS system REQUIRED)

include(cmake/flags.cmake)

if (WITH_TLS)
    find_package(OpenSSL)

    add_definitions(/DCPPHTTPLIB_OPENSSL_SUPPORT)

    if (OPENSSL_FOUND)
        include_directories(${OPENSSL_INCLUDE_DIR})
        set(SOURCES_SSL_TLS src/net/BoostTlsConnection.cpp)
    else()
        message(FATAL_ERROR "OpenSSL NOT found: use `-DWITH_TLS=OFF` to build without TLS support")
    endif()
else()
    add_definitions(/DXMRIG_NO_TLS)
endif()

if (WITH_LIBCPUID)
    add_subdirectory(src/3rdparty/libcpuid)

    include_directories(src/3rdparty/libcpuid)
    set(CPUID_LIB cpuid)
    set(SOURCES_CPUID src/Cpu_cpuid.cpp)
else()
    add_definitions(/DXMRIG_NO_LIBCPUID)

    if (XMRIG_ARM)
        set(SOURCES_CPUID src/Cpu_arm.cpp)
    else()
        set(SOURCES_CPUID src/Cpu_stub.cpp)
    endif()
endif()

CHECK_INCLUDE_FILE (syslog.h HAVE_SYSLOG_H)
if (HAVE_SYSLOG_H)
    add_definitions(/DHAVE_SYSLOG_H)
    set(SOURCES_SYSLOG src/log/SysLog.h src/log/SysLog.cpp)
endif()

if (WITH_HTTPD)
    find_package(MHD)

    if (MHD_FOUND)
        include_directories(${MHD_INCLUDE_DIRS})
        set(HTTPD_SOURCES src/api/Httpd.h src/api/Httpd.cpp)
    else()
        message(FATAL_ERROR "microhttpd NOT found: use `-DWITH_HTTPD=OFF` to build without http deamon support")
    endif()
else()
    add_definitions(/DXMRIG_NO_HTTPD)
    add_definitions(/DXMRIG_NO_API)
endif()

if (WITH_CC_SERVER)
    find_package(MHD)

    if (MHD_FOUND)
        include_directories(${MHD_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "microhttpd NOT found: use `-DWITH_CC_SERVER=OFF` to build without CC Server support")
    endif()

    set(SOURCES_CC_SERVER
            src/cc/CCServer.cpp
            src/cc/Service.cpp
            src/cc/Summary.cpp
            src/cc/Httpd.cpp
            src/cc/XMRigCC.cpp
            )
endif()

if (WITH_CC_CLIENT)
    set(SOURCES_CC_CLIENT
            src/cc/CCClient.cpp)
endif()

if (WITH_CC_SERVER OR WITH_CC_CLIENT)
    set(SOURCES_CC_COMMON
            src/cc/ControlCommand.cpp
            src/cc/ClientStatus.cpp)
else()
    add_definitions(/DXMRIG_NO_CC)
endif()

if (BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS " -static")
endif()

include_directories(src)
include_directories(src/3rdparty)
include_directories(${UV_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

add_library(xmrig_common STATIC ${SOURCES_COMMON})
add_library(xmrig_os_dependencies STATIC ${SOURCES_OS} ${SOURCES_SYSLOG})
add_library(xmrig_cpuid STATIC ${SOURCES_CPUID})

if (WITH_TLS)
    add_library(xmrig_tls STATIC ${SOURCES_SSL_TLS})
endif (WITH_TLS)

if (WITH_CC_SERVER OR WITH_CC_CLIENT)
    add_library(xmrig_cc_common STATIC ${SOURCES_CC_COMMON})
endif (WITH_CC_SERVER OR WITH_CC_CLIENT)

add_executable(xmrigMiner ${SOURCES} ${SOURCES_CRYPTO} ${HTTPD_SOURCES} ${SOURCES_CC_CLIENT} res/app.rc)
set_target_properties(xmrigMiner PROPERTIES OUTPUT_NAME ${MINER_EXECUTABLE_NAME})

target_link_libraries(xmrigMiner xmrig_common xmrig_os_dependencies xmrig_cpuid ${Boost_LIBRARIES}
        ${UV_LIBRARIES} ${MHD_LIBRARY} ${EXTRA_LIBS} ${CPUID_LIB})

if (WITH_CC_CLIENT)
    target_link_libraries(xmrigMiner xmrig_cc_common)
endif (WITH_CC_CLIENT)

if (WITH_TLS)
    target_link_libraries(xmrigMiner xmrig_tls ${OPENSSL_LIBRARIES} ${EXTRA_LIBS})
    target_link_libraries(xmrigMiner xmrig_tls ${OPENSSL_LIBRARIES} ${EXTRA_LIBS})
endif (WITH_TLS)

add_executable(xmrigDaemon src/cc/XMRigd.cpp res/app.rc)
set_target_properties(xmrigDaemon PROPERTIES OUTPUT_NAME ${DAEMON_EXECUTABLE_NAME})

if (WITH_CC_SERVER AND MHD_FOUND)
    add_library(xmrig_common_cc STATIC ${SOURCES_COMMON})
    add_executable(xmrigCCServer ${SOURCES_CC_SERVER} res/app.rc)
    target_link_libraries(xmrigCCServer
            xmrig_common_cc xmrig_os_dependencies xmrig_cpuid xmrig_cc_common
            ${UV_LIBRARIES} ${MHD_LIBRARY} ${EXTRA_LIBS} ${CPUID_LIB})

    if (WITH_TLS)
	    target_link_libraries(xmrigCCServer xmrig_tls ${OPENSSL_LIBRARIES})
    endif (WITH_TLS)

    set_target_properties(xmrig_common_cc PROPERTIES COMPILE_FLAGS "-DXMRIG_CC_SERVER ${SHARED_FLAGS}")
    set_target_properties(xmrigCCServer PROPERTIES COMPILE_FLAGS "-DXMRIG_CC_SERVER ${SHARED_FLAGS}")
endif()

add_subdirectory(test EXCLUDE_FROM_ALL)
