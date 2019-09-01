if (WITH_OPENCL)
    add_definitions(/DCL_TARGET_OPENCL_VERSION=200)
    add_definitions(/DCL_USE_DEPRECATED_OPENCL_1_2_APIS)
    add_definitions(/DXMRIG_FEATURE_OPENCL)

    set(HEADERS_BACKEND_OPENCL
        src/backend/opencl/cl/OclSource.h
        src/backend/opencl/interfaces/IOclRunner.h
        src/backend/opencl/kernels/Cn0Kernel.h
        src/backend/opencl/kernels/Cn1Kernel.h
        src/backend/opencl/OclBackend.h
        src/backend/opencl/OclCache.h
        src/backend/opencl/OclConfig.h
        src/backend/opencl/OclLaunchData.h
        src/backend/opencl/OclThread.h
        src/backend/opencl/OclThreads.h
        src/backend/opencl/OclWorker.h
        src/backend/opencl/runners/OclBaseRunner.h
        src/backend/opencl/runners/OclCnRunner.h
        src/backend/opencl/wrappers/OclContext.h
        src/backend/opencl/wrappers/OclDevice.h
        src/backend/opencl/wrappers/OclError.h
        src/backend/opencl/wrappers/OclKernel.h
        src/backend/opencl/wrappers/OclLib.h
        src/backend/opencl/wrappers/OclPlatform.h
        src/backend/opencl/wrappers/OclVendor.h
       )

    set(SOURCES_BACKEND_OPENCL
        src/backend/opencl/cl/OclSource.cpp
        src/backend/opencl/kernels/Cn0Kernel.cpp
        src/backend/opencl/kernels/Cn1Kernel.cpp
        src/backend/opencl/OclBackend.cpp
        src/backend/opencl/OclCache.cpp
        src/backend/opencl/OclConfig.cpp
        src/backend/opencl/OclLaunchData.cpp
        src/backend/opencl/OclThread.cpp
        src/backend/opencl/OclThreads.cpp
        src/backend/opencl/OclWorker.cpp
        src/backend/opencl/runners/OclBaseRunner.cpp
        src/backend/opencl/runners/OclCnRunner.cpp
        src/backend/opencl/wrappers/OclContext.cpp
        src/backend/opencl/wrappers/OclDevice.cpp
        src/backend/opencl/wrappers/OclError.cpp
        src/backend/opencl/wrappers/OclKernel.cpp
        src/backend/opencl/wrappers/OclLib.cpp
        src/backend/opencl/wrappers/OclPlatform.cpp
       )

   if (WIN32)
       list(APPEND SOURCES_BACKEND_OPENCL src/backend/opencl/OclCache_win.cpp)
   else()
       list(APPEND SOURCES_BACKEND_OPENCL src/backend/opencl/OclCache_unix.cpp)
   endif()

   if (WITH_RANDOMX)
       list(APPEND HEADERS_BACKEND_OPENCL src/backend/opencl/runners/OclRxRunner.h)
       list(APPEND SOURCES_BACKEND_OPENCL src/backend/opencl/runners/OclRxRunner.cpp)
   endif()

   if (WITH_STRICT_CACHE)
       add_definitions(/DXMRIG_STRICT_OPENCL_CACHE)
   else()
       remove_definitions(/DXMRIG_STRICT_OPENCL_CACHE)
   endif()
else()
    remove_definitions(/DXMRIG_FEATURE_OPENCL)

    set(HEADERS_BACKEND_OPENCL "")
    set(SOURCES_BACKEND_OPENCL "")
endif()
