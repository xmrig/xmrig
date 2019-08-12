if (WITH_OPENCL)
    add_definitions(/DCL_TARGET_OPENCL_VERSION=200)
    add_definitions(/DCL_USE_DEPRECATED_OPENCL_1_2_APIS)
    add_definitions(/DXMRIG_FEATURE_OPENCL)

    set(HEADERS_BACKEND_OPENCL
        src/backend/opencl/OclError.h
        src/backend/opencl/OclLib.h
       )

    set(SOURCES_BACKEND_OPENCL
        src/backend/opencl/OclError.cpp
        src/backend/opencl/OclLib.cpp
       )
else()
    remove_definitions(/DXMRIG_FEATURE_OPENCL)

    set(HEADERS_BACKEND_OPENCL "")
    set(SOURCES_BACKEND_OPENCL "")
endif()
