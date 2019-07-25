set(HEADERS_BACKEND_CPU
    src/backend/cpu/Cpu.h
    src/backend/cpu/CpuBackend.h
    src/backend/cpu/CpuConfig.h
    src/backend/cpu/CpuLaunchData.cpp
    src/backend/cpu/CpuThread.h
    src/backend/cpu/CpuWorker.h
    src/backend/cpu/interfaces/ICpuInfo.h
   )

set(SOURCES_BACKEND_CPU
    src/backend/cpu/Cpu.cpp
    src/backend/cpu/CpuBackend.cpp
    src/backend/cpu/CpuConfig.cpp
    src/backend/cpu/CpuLaunchData.h
    src/backend/cpu/CpuThread.cpp
    src/backend/cpu/CpuWorker.cpp
   )


if (WITH_HWLOC)
    find_package(HWLOC REQUIRED)

    set(WITH_LIBCPUID OFF)

    include_directories(${HWLOC_INCLUDE_DIR})

    remove_definitions(/DXMRIG_FEATURE_LIBCPUID)
    add_definitions(/DXMRIG_FEATURE_HWLOC)

    if (HWLOC_DEBUG)
        add_definitions(/DXMRIG_HWLOC_DEBUG)
    endif()

    set(CPUID_LIB "")
    set(SOURCES_CPUID
        src/backend/cpu/platform/BasicCpuInfo.cpp
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

    if (XMRIG_ARM)
        set(SOURCES_CPUID ${SOURCES_CPUID} src/backend/cpu/platform/BasicCpuInfo_arm.cpp)
    else()
        set(SOURCES_CPUID ${SOURCES_CPUID} src/backend/cpu/platform/BasicCpuInfo.cpp)
    endif()
endif()
