if (WITH_CUDA)
    add_definitions(/DXMRIG_FEATURE_CUDA)

    set(HEADERS_BACKEND_CUDA
        src/backend/cuda/CudaBackend.h
        src/backend/cuda/CudaConfig.h
        src/backend/cuda/CudaConfig_gen.h
        src/backend/cuda/CudaThread.h
        src/backend/cuda/CudaThreads.h
       )

    set(SOURCES_BACKEND_CUDA
        src/backend/cuda/CudaBackend.cpp
        src/backend/cuda/CudaConfig.cpp
        src/backend/cuda/CudaThread.cpp
        src/backend/cuda/CudaThreads.cpp
       )
else()
    remove_definitions(/DXMRIG_FEATURE_CUDA)

    set(HEADERS_BACKEND_CUDA "")
    set(SOURCES_BACKEND_CUDA "")
endif()
