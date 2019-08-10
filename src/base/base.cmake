set(HEADERS_BASE
    src/base/io/Console.h
    src/base/io/json/Json.h
    src/base/io/json/JsonChain.h
    src/base/io/json/JsonRequest.h
    src/base/io/log/backends/ConsoleLog.h
    src/base/io/log/backends/FileLog.h
    src/base/io/log/Log.h
    src/base/io/Watcher.h
    src/base/kernel/Base.h
    src/base/kernel/config/BaseConfig.h
    src/base/kernel/config/BaseTransform.h
    src/base/kernel/Entry.h
    src/base/kernel/interfaces/IBaseListener.h
    src/base/kernel/interfaces/IClient.h
    src/base/kernel/interfaces/IClientListener.h
    src/base/kernel/interfaces/IConfig.h
    src/base/kernel/interfaces/IConfigListener.h
    src/base/kernel/interfaces/IConfigTransform.h
    src/base/kernel/interfaces/IConsoleListener.h
    src/base/kernel/interfaces/IDnsListener.h
    src/base/kernel/interfaces/ILineListener.h
    src/base/kernel/interfaces/ILogBackend.h
    src/base/kernel/interfaces/ISignalListener.h
    src/base/kernel/interfaces/IStrategy.h
    src/base/kernel/interfaces/IStrategyListener.h
    src/base/kernel/interfaces/ITimerListener.h
    src/base/kernel/interfaces/IWatcherListener.h
    src/base/kernel/Platform.h
    src/base/kernel/Process.h
    src/base/kernel/Signals.h
    src/base/net/dns/Dns.h
    src/base/net/dns/DnsRecord.h
    src/base/net/http/Http.h
    src/base/net/stratum/BaseClient.h
    src/base/net/stratum/Client.h
    src/base/net/stratum/Job.h
    src/base/net/stratum/Pool.h
    src/base/net/stratum/Pools.h
    src/base/net/stratum/strategies/FailoverStrategy.h
    src/base/net/stratum/strategies/SinglePoolStrategy.h
    src/base/net/stratum/SubmitResult.h
    src/base/net/tools/RecvBuf.h
    src/base/net/tools/Storage.h
    src/base/tools/Arguments.h
    src/base/tools/Baton.h
    src/base/tools/Buffer.h
    src/base/tools/Chrono.h
    src/base/tools/Handle.h
    src/base/tools/String.h
    src/base/tools/Timer.h
   )

set(SOURCES_BASE
    src/base/io/Console.cpp
    src/base/io/json/Json.cpp
    src/base/io/json/JsonChain.cpp
    src/base/io/json/JsonRequest.cpp
    src/base/io/log/backends/ConsoleLog.cpp
    src/base/io/log/backends/FileLog.cpp
    src/base/io/log/Log.cpp
    src/base/io/Watcher.cpp
    src/base/kernel/Base.cpp
    src/base/kernel/config/BaseConfig.cpp
    src/base/kernel/config/BaseTransform.cpp
    src/base/kernel/Entry.cpp
    src/base/kernel/Platform.cpp
    src/base/kernel/Process.cpp
    src/base/kernel/Signals.cpp
    src/base/net/dns/Dns.cpp
    src/base/net/dns/DnsRecord.cpp
    src/base/net/http/Http.cpp
    src/base/net/stratum/BaseClient.cpp
    src/base/net/stratum/Client.cpp
    src/base/net/stratum/Job.cpp
    src/base/net/stratum/Pool.cpp
    src/base/net/stratum/Pools.cpp
    src/base/net/stratum/strategies/FailoverStrategy.cpp
    src/base/net/stratum/strategies/SinglePoolStrategy.cpp
    src/base/tools/Arguments.cpp
    src/base/tools/Buffer.cpp
    src/base/tools/String.cpp
    src/base/tools/Timer.cpp
   )


if (WIN32)
    set(SOURCES_OS
        src/base/io/json/Json_win.cpp
        src/base/kernel/Platform_win.cpp
        )
elseif (APPLE)
    set(SOURCES_OS
        src/base/io/json/Json_unix.cpp
        src/base/kernel/Platform_mac.cpp
        )
else()
    set(SOURCES_OS
        src/base/io/json/Json_unix.cpp
        src/base/kernel//Platform_unix.cpp
        )
endif()


if (NOT WIN32)
    CHECK_INCLUDE_FILE (syslog.h HAVE_SYSLOG_H)
    if (HAVE_SYSLOG_H)
        add_definitions(/DHAVE_SYSLOG_H)
        set(SOURCES_SYSLOG src/base/io/log/backends/SysLog.h src/base/io/log/backends/SysLog.cpp)
    endif()
endif()


if (WITH_HTTP)
    set(HEADERS_BASE_HTTP
        src/3rdparty/http-parser/http_parser.h
        src/base/kernel/interfaces/IHttpListener.h
        src/base/kernel/interfaces/IJsonReader.h
        src/base/kernel/interfaces/ITcpServerListener.h
        src/base/net/http/HttpApiResponse.h
        src/base/net/http/HttpClient.h
        src/base/net/http/HttpContext.h
        src/base/net/http/HttpData.h
        src/base/net/http/HttpResponse.h
        src/base/net/http/HttpServer.h
        src/base/net/stratum/DaemonClient.h
        src/base/net/tools/TcpServer.h
        )

    set(SOURCES_BASE_HTTP
        src/3rdparty/http-parser/http_parser.c
        src/base/net/http/HttpApiResponse.cpp
        src/base/net/http/HttpClient.cpp
        src/base/net/http/HttpContext.cpp
        src/base/net/http/HttpResponse.cpp
        src/base/net/http/HttpServer.cpp
        src/base/net/stratum/DaemonClient.cpp
        src/base/net/tools/TcpServer.cpp
        )

    add_definitions(/DXMRIG_FEATURE_HTTP)
    add_definitions(/DXMRIG_FEATURE_API)
else()
    set(HEADERS_BASE_HTTP "")
    set(SOURCES_BASE_HTTP "")
    remove_definitions(/DXMRIG_FEATURE_HTTP)
    remove_definitions(/DXMRIG_FEATURE_API)
endif()
