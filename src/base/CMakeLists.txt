cmake_minimum_required(VERSION 3.1)
project(base LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(UV REQUIRED)
include_directories(${UV_INCLUDE_DIR})
set(LIBS ${UV_LIBRARIES})

set(HEADERS
    3rdparty/epee/span.h
    crypto/Algorithm.h
    crypto/Coin.h
    crypto/keccak.h
    crypto/sha3.h
    io/Async.h
    io/Console.h
    io/Env.h
    io/json/Json.h
    io/json/JsonChain.h
    io/json/JsonRequest.h
    io/log/backends/ConsoleLog.h
    io/log/backends/FileLog.h
    io/log/FileLogWriter.h
    io/log/Log.h
    io/log/Tags.h
    io/Signals.h
    io/Watcher.h
    kernel/Base.h
    kernel/config/BaseConfig.h
    kernel/config/BaseTransform.h
    kernel/config/Title.h
    kernel/constants.h
    kernel/Entry.h
    kernel/interfaces/IAsyncListener.h
    kernel/interfaces/IBaseListener.h
    kernel/interfaces/IClient.h
    kernel/interfaces/IClientListener.h
    kernel/interfaces/IConfig.h
    kernel/interfaces/IConfigListener.h
    kernel/interfaces/IConsoleListener.h
    kernel/interfaces/IDnsBackend.h
    kernel/interfaces/IDnsListener.h
    kernel/interfaces/IHttpListener.h
    kernel/interfaces/IJsonReader.h
    kernel/interfaces/ILineListener.h
    kernel/interfaces/ILogBackend.h
    kernel/interfaces/ISignalListener.h
    kernel/interfaces/IStrategy.h
    kernel/interfaces/IStrategyListener.h
    kernel/interfaces/ITcpServerListener.h
    kernel/interfaces/ITimerListener.h
    kernel/interfaces/IWatcherListener.h
    kernel/Platform.h
    kernel/Process.h
    kernel/Versions.h
    net/dns/Dns.h
    net/dns/DnsConfig.h
    net/dns/DnsRecord.h
    net/dns/DnsRecords.h
    net/dns/DnsRequest.h
    net/dns/DnsUvBackend.h
    net/http/Http.h
    net/stratum/BaseClient.h
    net/stratum/Client.h
    net/stratum/Job.h
    net/stratum/NetworkState.h
    net/stratum/Pool.h
    net/stratum/Pools.h
    net/stratum/ProxyUrl.h
    net/stratum/Socks5.h
    net/stratum/strategies/FailoverStrategy.h
    net/stratum/strategies/SinglePoolStrategy.h
    net/stratum/SubmitResult.h
    net/stratum/Url.h
    net/tools/LineReader.h
    net/tools/MemPool.h
    net/tools/NetBuffer.h
    net/tools/Storage.h
    tools/Arguments.h
    tools/Baton.h
    tools/bswap_64.h
    tools/Buffer.h
    tools/Chrono.h
    tools/Cvt.h
    tools/Handle.h
    tools/Object.h
    tools/Span.h
    tools/String.h
    tools/Timer.h
    )

set(SOURCES
    3rdparty/fmt/format.cc
    crypto/Algorithm.cpp
    crypto/Coin.cpp
    crypto/keccak.cpp
    crypto/sha3.cpp
    io/Async.cpp
    io/Console.cpp
    io/Env.cpp
    io/json/Json.cpp
    io/json/JsonChain.cpp
    io/json/JsonRequest.cpp
    io/log/backends/ConsoleLog.cpp
    io/log/backends/FileLog.cpp
    io/log/FileLogWriter.cpp
    io/log/Log.cpp
    io/log/Tags.cpp
    io/Signals.cpp
    io/Watcher.cpp
    kernel/Base.cpp
    kernel/config/BaseConfig.cpp
    kernel/config/BaseTransform.cpp
    kernel/config/Title.cpp
    kernel/Entry.cpp
    kernel/Platform.cpp
    kernel/Process.cpp
    kernel/Versions.cpp
    net/dns/Dns.cpp
    net/dns/DnsConfig.cpp
    net/dns/DnsRecord.cpp
    net/dns/DnsRecords.cpp
    net/dns/DnsUvBackend.cpp
    net/http/Http.cpp
    net/stratum/BaseClient.cpp
    net/stratum/Client.cpp
    net/stratum/Job.cpp
    net/stratum/NetworkState.cpp
    net/stratum/Pool.cpp
    net/stratum/Pools.cpp
    net/stratum/ProxyUrl.cpp
    net/stratum/Socks5.cpp
    net/stratum/strategies/FailoverStrategy.cpp
    net/stratum/strategies/SinglePoolStrategy.cpp
    net/stratum/Url.cpp
    net/tools/LineReader.cpp
    net/tools/NetBuffer.cpp
    tools/Arguments.cpp
    tools/Cvt.cpp
    tools/String.cpp
    tools/Timer.cpp
    )

if (XMRIG_OS_WIN)
    list(APPEND SOURCES
        io/json/Json_win.cpp
        kernel/Platform_win.cpp
        kernel/Process_win.cpp
        )

    list(APPEND LIBS ws2_32 psapi iphlpapi userenv)
elseif (XMRIG_OS_APPLE)
    list(APPEND SOURCES
        io/json/Json_unix.cpp
        kernel/Platform_mac.cpp
        kernel/Process_unix.cpp
        )

    find_library(IOKIT_LIBRARY IOKit)
    find_library(CORESERVICES_LIBRARY CoreServices)
    list(APPEND LIBS ${IOKIT_LIBRARY} ${CORESERVICES_LIBRARY})
else()
    list(APPEND SOURCES
        io/json/Json_unix.cpp
        kernel/Platform_unix.cpp
        kernel/Process_unix.cpp
        )

    list(APPEND LIBS pthread)

    if (XMRIG_OS_ANDROID)
        list(APPEND LIBS rt dl log)
    elseif (XMRIG_OS_LINUX)
        list(APPEND LIBS rt dl)
    elseif (XMRIG_OS_FREEBSD)
        list(APPEND LIBS kvm)
    endif()
endif()

if (HAVE_SYSLOG_H)
    list(APPEND HEADERS io/log/backends/SysLog.h)
    list(APPEND SOURCES io/log/backends/SysLog.cpp)
endif()

if (WITH_HTTP)
    list(APPEND HEADERS
        3rdparty/llhttp/llhttp.h
        api/Api.h
        api/Httpd.h
        api/interfaces/IApiRequest.h
        api/requests/ApiRequest.h
        api/requests/HttpApiRequest.h
        net/http/Fetch.h
        net/http/HttpApiResponse.h
        net/http/HttpClient.h
        net/http/HttpContext.h
        net/http/HttpData.h
        net/http/HttpListener.h
        net/http/HttpResponse.h
        net/stratum/DaemonClient.h
        net/stratum/SelfSelectClient.h
        net/tools/TcpServer.h
        )

    list(APPEND SOURCES
        3rdparty/llhttp/api.c
        3rdparty/llhttp/http.c
        3rdparty/llhttp/llhttp.c
        api/Api.cpp
        api/Httpd.cpp
        api/requests/ApiRequest.cpp
        api/requests/HttpApiRequest.cpp
        net/http/Fetch.cpp
        net/http/HttpApiResponse.cpp
        net/http/HttpClient.cpp
        net/http/HttpContext.cpp
        net/http/HttpData.cpp
        net/http/HttpListener.cpp
        net/http/HttpResponse.cpp
        net/stratum/DaemonClient.cpp
        net/stratum/SelfSelectClient.cpp
        net/tools/TcpServer.cpp
        )
endif()

if (WITH_TLS)
    list(APPEND HEADERS
        net/stratum/Tls.h
        net/tls/ServerTls.h
        net/tls/TlsConfig.h
        net/tls/TlsContext.h
        net/tls/TlsGen.h
        )

    list(APPEND SOURCES
        net/stratum/Tls.cpp
        net/tls/ServerTls.cpp
        net/tls/TlsConfig.cpp
        net/tls/TlsContext.cpp
        net/tls/TlsGen.cpp
        )

    if (WITH_HTTP)
        list(APPEND HEADERS
            net/https/HttpsClient.h
            net/https/HttpsContext.h
            net/https/HttpsServer.h
            )

        list(APPEND SOURCES
            net/https/HttpsClient.cpp
            net/https/HttpsContext.cpp
            net/https/HttpsServer.cpp
            )
    endif()
endif()

if (WITH_HTTP AND NOT WITH_TLS)
    list(APPEND HEADERS net/http/HttpServer.h)
    list(APPEND SOURCES net/http/HttpServer.cpp)
endif()

if (WITH_CRYPTONOTE)
    list(APPEND HEADERS
        tools/cryptonote/BlobReader.h
        tools/cryptonote/BlockTemplate.h
        tools/cryptonote/umul128.h
        tools/cryptonote/WalletAddress.h
        )

    list(APPEND SOURCES
        tools/cryptonote/BlockTemplate.cpp
        tools/cryptonote/WalletAddress.cpp
        )
endif()

if (WITH_CRYPTO_OPS)
    list(APPEND HEADERS
        3rdparty/cryptonote/crypto-ops.h
        tools/cryptonote/Signatures.h
        )

    list(APPEND SOURCES
        3rdparty/cryptonote/crypto-ops.c
        3rdparty/cryptonote/crypto-ops-data.c
        tools/cryptonote/Signatures.cpp
        )
endif()

if (WITH_KAWPOW)
    list(APPEND HEADERS
        3rdparty/libethash/endian.h
        net/stratum/AutoClient.h
        net/stratum/EthStratumClient.h
        )

    list(APPEND SOURCES
        net/stratum/AutoClient.cpp
        net/stratum/EthStratumClient.cpp
        )
endif()

if (WITH_BENCHMARK)
    list(APPEND HEADERS
        net/stratum/benchmark/BenchClient.h
        net/stratum/benchmark/BenchConfig.h
        )

    list(APPEND SOURCES
        net/stratum/benchmark/BenchClient.cpp
        net/stratum/benchmark/BenchConfig.cpp
        )
endif()

if (WITH_HWLOC)
    list(APPEND SOURCES kernel/Platform_hwloc.cpp)
endif()

add_library(${PROJECT_NAME} ${HEADERS} ${SOURCES})

if (WITH_TLS)
    target_include_directories(${PROJECT_NAME} PUBLIC ${OPENSSL_INCLUDE_DIR})
    list(INSERT LIBS 1 ${OPENSSL_LIBRARIES})

    if (XMRIG_OS_WIN)
        list(APPEND LIBS crypt32)
    endif()
endif()

if (WITH_HWLOC)
    target_include_directories(${PROJECT_NAME} PUBLIC ${HWLOC_INCLUDE_DIR})
    list(APPEND LIBS ${HWLOC_LIBRARY})
endif()

if (WITH_SODIUM)
    target_include_directories(${PROJECT_NAME} PUBLIC ${SODIUM_INCLUDE_DIRS})
    list(APPEND LIBS ${SODIUM_LIBRARIES})
endif()

target_link_libraries(${PROJECT_NAME} ${LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC ${UV_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
