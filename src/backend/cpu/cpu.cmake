set(HEADERS_CPU
    src/backend/cpu/Cpu.h
    src/backend/cpu/CpuConfig.h
    src/backend/cpu/interfaces/ICpuInfo.h
   )

set(SOURCES_CPU
    src/backend/cpu/Cpu.cpp
    src/backend/cpu/CpuConfig.cpp
   )


if (WITH_LIBCPUID)
    add_subdirectory(src/3rdparty/libcpuid)
    include_directories(src/3rdparty/libcpuid)
    add_definitions(/DXMRIG_FEATURE_LIBCPUID)

    set(CPUID_LIB cpuid)
    set(SOURCES_CPUID src/backend/cpu/platform/AdvancedCpuInfo.h src/backend/cpu/platform/AdvancedCpuInfo.cpp src/backend/cpu/Cpu.cpp)
else()
    remove_definitions(/DXMRIG_FEATURE_LIBCPUID)
    set(SOURCES_CPUID src/backend/cpu/platform/BasicCpuInfo.h src/backend/cpu/Cpu.cpp)

    if (XMRIG_ARM)
        set(SOURCES_CPUID ${SOURCES_CPUID} src/backend/cpu/platform/BasicCpuInfo_arm.cpp)
    else()
        set(SOURCES_CPUID ${SOURCES_CPUID} src/backend/cpu/platform/BasicCpuInfo.cpp)
    endif()
endif()
