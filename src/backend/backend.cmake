include (src/backend/cpu/cpu.cmake)


set(HEADERS_BACKEND
    "${HEADERS_CPU}"
    src/backend/Threads.h
   )

set(SOURCES_BACKEND
    "${SOURCES_CPU}"
    src/backend/Threads.cpp
   )
