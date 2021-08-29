set(HEADERS_BACKEND_CPU
    src/backend/cpu/Cpu.h
    src/backend/cpu/CpuBackend.h
    src/backend/cpu/CpuConfig_gen.h
    src/backend/cpu/CpuConfig.h
    src/backend/cpu/CpuLaunchData.cpp
    src/backend/cpu/CpuThread.h
    src/backend/cpu/CpuThreads.h
    src/backend/cpu/CpuWorker.h
    src/backend/cpu/interfaces/ICpuInfo.h
    src/backend/cpu/platform/BasicCpuInfo.h
   )

set(SOURCES_BACKEND_CPU
    src/backend/cpu/Cpu.cpp
    src/backend/cpu/CpuBackend.cpp
    src/backend/cpu/CpuConfig.cpp
    src/backend/cpu/CpuLaunchData.h
    src/backend/cpu/CpuThread.cpp
    src/backend/cpu/CpuThreads.cpp
    src/backend/cpu/CpuWorker.cpp
   )

if (WITH_HWLOC)
    if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        add_subdirectory(src/3rdparty/hwloc)
        include_directories(src/3rdparty/hwloc/include)
        set(CPUID_LIB hwloc)
    else()
        find_package(HWLOC REQUIRED)
        include_directories(${HWLOC_INCLUDE_DIR})
        set(CPUID_LIB ${HWLOC_LIBRARY})
    endif()

    add_definitions(/DXMRIG_FEATURE_HWLOC)

    if (HWLOC_DEBUG)
        add_definitions(/DXMRIG_HWLOC_DEBUG)
    endif()

    list(APPEND HEADERS_BACKEND_CPU src/backend/cpu/platform/HwlocCpuInfo.h)
    list(APPEND SOURCES_BACKEND_CPU src/backend/cpu/platform/HwlocCpuInfo.cpp)
else()
    remove_definitions(/DXMRIG_FEATURE_HWLOC)

    set(CPUID_LIB "")
endif()

if (XMRIG_ARM)
    list(APPEND SOURCES_BACKEND_CPU src/backend/cpu/platform/BasicCpuInfo_arm.cpp)

    if (XMRIG_OS_UNIX)
        list(APPEND SOURCES_BACKEND_CPU src/backend/cpu/platform/lscpu_arm.cpp)
    endif()
else()
    list(APPEND SOURCES_BACKEND_CPU src/backend/cpu/platform/BasicCpuInfo.cpp)
endif()
