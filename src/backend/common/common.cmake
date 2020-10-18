set(HEADERS_BACKEND_COMMON
    src/backend/common/Hashrate.h
    src/backend/common/HashrateInterpolator.h
    src/backend/common/Tags.h
    src/backend/common/interfaces/IBackend.h
    src/backend/common/interfaces/IRxListener.h
    src/backend/common/interfaces/IRxStorage.h
    src/backend/common/interfaces/IWorker.h
    src/backend/common/misc/PciTopology.h
    src/backend/common/Thread.h
    src/backend/common/Threads.h
    src/backend/common/Worker.h
    src/backend/common/WorkerJob.h
    src/backend/common/Workers.h
   )

set(SOURCES_BACKEND_COMMON
    src/backend/common/Hashrate.cpp
    src/backend/common/HashrateInterpolator.cpp
    src/backend/common/Threads.cpp
    src/backend/common/Worker.cpp
    src/backend/common/Workers.cpp
   )

if (WITH_RANDOMX AND WITH_BENCHMARK)
    list(APPEND HEADERS_BACKEND_COMMON src/backend/common/Benchmark.h)
    list(APPEND SOURCES_BACKEND_COMMON src/backend/common/Benchmark.cpp)
endif()
