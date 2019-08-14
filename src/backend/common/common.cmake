set(HEADERS_BACKEND_COMMON
    src/backend/common/interfaces/IBackend.h
    src/backend/common/interfaces/IThread.h
    src/backend/common/interfaces/IWorker.h
    src/backend/common/Hashrate.h
    src/backend/common/Thread.h
    src/backend/common/Threads.h
    src/backend/common/Worker.h
    src/backend/common/Workers.h
    src/backend/common/WorkerJob.h
   )

set(SOURCES_BACKEND_COMMON
    src/backend/common/Hashrate.cpp
    src/backend/common/Threads.cpp
    src/backend/common/Worker.cpp
    src/backend/common/Workers.cpp
   )
