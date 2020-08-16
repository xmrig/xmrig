set(HEADERS_BACKEND_CPU
    src/backend/cpu/Cpu.h
    src/backend/cpu/CpuBackend.h
    src/backend/cpu/CpuConfig.h
    src/backend/cpu/CpuConfig_gen.h
    src/backend/cpu/CpuLaunchData.cpp
    src/backend/cpu/CpuThread.h
    src/backend/cpu/CpuThreads.h
    src/backend/cpu/CpuWorker.h
    src/backend/cpu/interfaces/ICpuInfo.h
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

    set(WITH_LIBCPUID OFF)

    remove_definitions(/DXMRIG_FEATURE_LIBCPUID)
    add_definitions(/DXMRIG_FEATURE_HWLOC)

    if (HWLOC_DEBUG)
        add_definitions(/DXMRIG_HWLOC_DEBUG)
    endif()

    set(SOURCES_CPUID
        src/backend/cpu/platform/BasicCpuInfo.h
        src/backend/cpu/platform/HwlocCpuInfo.cpp
        src/backend/cpu/platform/HwlocCpuInfo.h
        )
elseif (WITH_LIBCPUID)
    set(WITH_HWLOC OFF)

    add_subdirectory(src/3rdparty/libcpuid)
    include_directories(src/3rdparty/libcpuid)

    add_definitions(/DXMRIG_FEATURE_LIBCPUID)
    remove_definitions(/DXMRIG_FEATURE_HWLOC)

    set(CPUID_LIB cpuid)
    set(SOURCES_CPUID
        src/backend/cpu/platform/AdvancedCpuInfo.cpp
        src/backend/cpu/platform/AdvancedCpuInfo.h
        )
else()
    remove_definitions(/DXMRIG_FEATURE_LIBCPUID)
    remove_definitions(/DXMRIG_FEATURE_HWLOC)

    set(CPUID_LIB "")
    set(SOURCES_CPUID
        src/backend/cpu/platform/BasicCpuInfo.h
        )
endif()

if (XMRIG_ARM)
    list(APPEND SOURCES_CPUID src/backend/cpu/platform/BasicCpuInfo_arm.cpp)

    if (XMRIG_OS_UNIX)
        list(APPEND SOURCES_CPUID src/backend/cpu/platform/lscpu_arm.cpp)
    endif()
else()
    list(APPEND SOURCES_CPUID src/backend/cpu/platform/BasicCpuInfo.cpp)
endif()
