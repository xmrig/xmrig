cmake_minimum_required(VERSION 3.0)
project(xmrig)

set(HEADERS
    src/App.h
    src/interfaces/IClientListener.h
    src/net/Client.h
    src/net/Job.h
    src/net/Network.h
    src/net/Url.h
    src/Options.h
    src/Console.h
    src/version.h
   )

set(SOURCES
    src/App.cpp
    src/net/Client.cpp
    src/net/Job.cpp
    src/net/Network.cpp
    src/net/Url.cpp
    src/Options.cpp
    src/Console.cpp
    src/xmrig.cpp
   )

if (WIN32)
    set(SOURCES_OS
        res/app.rc
        src/3rdparty/winansi.cpp
        src/3rdparty/winansi.h
        src/net/Network_win.cpp
        )

    set(EXTRA_LIBS ws2_32 psapi iphlpapi userenv)
else()
    set(SOURCES_OS unix/cpu_unix.c unix/memory_unix.c unix/xmrig_unix.c)
    set(EXTRA_LIBS pthread)
endif()

add_definitions(/D_GNU_SOURCE)
add_definitions(/DUNICODE)
add_definitions(/DAPP_DEBUG)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(UV REQUIRED)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -funroll-loops -fvariable-expansion-in-unroller -ftree-loop-if-convert-stores -fmerge-all-constants -fbranch-target-load-optimize2")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -gdwarf-2")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fprofile-generate")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fprofile-use -fprofile-correction")

if (WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
endif()

include_directories(src)
include_directories(src/3rdparty)
include_directories(src/3rdparty/jansson)
include_directories(${UV_INCLUDE_DIR})

add_subdirectory(src/3rdparty/jansson)

add_executable(xmrig ${HEADERS} ${SOURCES} ${SOURCES_OS})
target_link_libraries(xmrig jansson ${UV_LIBRARIES} ${EXTRA_LIBS})
