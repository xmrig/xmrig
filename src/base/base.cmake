set(HEADERS_BASE
    src/base/io/Console.h
    src/base/io/Json.h
    src/base/io/log/backends/ConsoleLog.h
    src/base/io/log/backends/FileLog.h
    src/base/io/log/Log.h
    src/base/io/Watcher.h
    src/base/kernel/Entry.h
    src/base/kernel/interfaces/IClientListener.h
    src/base/kernel/interfaces/IConfigListener.h
    src/base/kernel/interfaces/IConsoleListener.h
    src/base/kernel/interfaces/IDnsListener.h
    src/base/kernel/interfaces/ILineListener.h
    src/base/kernel/interfaces/ILogBackend.h
    src/base/kernel/interfaces/ISignalListener.h
    src/base/kernel/interfaces/IStrategy.h
    src/base/kernel/interfaces/IStrategyListener.h
    src/base/kernel/interfaces/ITimerListener.h
    src/base/kernel/interfaces/IWatcherListener.h
    src/base/kernel/Process.h
    src/base/kernel/Signals.h
    src/base/net/dns/Dns.h
    src/base/net/dns/DnsRecord.h
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
    src/base/tools/Buffer.h
    src/base/tools/Chrono.h
    src/base/tools/Handle.h
    src/base/tools/String.h
    src/base/tools/Timer.h
   )

set(SOURCES_BASE
    src/base/io/Console.cpp
    src/base/io/Json.cpp
    src/base/io/log/backends/ConsoleLog.cpp
    src/base/io/log/backends/FileLog.cpp
    src/base/io/log/Log.cpp
    src/base/io/Watcher.cpp
    src/base/kernel/Entry.cpp
    src/base/kernel/Process.cpp
    src/base/kernel/Signals.cpp
    src/base/net/dns/Dns.cpp
    src/base/net/dns/DnsRecord.cpp
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
    set(SOURCES_OS src/base/io/Json_win.cpp)
else()
    set(SOURCES_OS src/base/io/Json_unix.cpp)
endif()


if (NOT WIN32)
    CHECK_INCLUDE_FILE (syslog.h HAVE_SYSLOG_H)
    if (HAVE_SYSLOG_H)
        add_definitions(/DHAVE_SYSLOG_H)
        set(SOURCES_SYSLOG src/base/log/backends/SysLog.h src/base/log/backends/SysLog.cpp)
    endif()
endif()
