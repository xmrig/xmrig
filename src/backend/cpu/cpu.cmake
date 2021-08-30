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
    list(APPEND HEADERS_BACKEND_CPU src/backend/cpu/platform/HwlocCpuInfo.h)
    list(APPEND SOURCES_BACKEND_CPU src/backend/cpu/platform/HwlocCpuInfo.cpp)
endif()

if (XMRIG_ARM)
    list(APPEND SOURCES_BACKEND_CPU src/backend/cpu/platform/BasicCpuInfo_arm.cpp)

    if (XMRIG_OS_UNIX)
        list(APPEND SOURCES_BACKEND_CPU src/backend/cpu/platform/lscpu_arm.cpp)
    endif()
else()
    list(APPEND SOURCES_BACKEND_CPU src/backend/cpu/platform/BasicCpuInfo.cpp)
endif()
