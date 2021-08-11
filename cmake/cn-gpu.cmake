if (WITH_CN_GPU AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    add_definitions(/DXMRIG_ALGO_CN_GPU)

    if (XMRIG_ARM)
        list(APPEND SOURCES_CRYPTO
            src/crypto/cn/gpu/cn_gpu_arm.cpp
        )

        if (CMAKE_CXX_COMPILER_ID MATCHES GNU OR CMAKE_CXX_COMPILER_ID MATCHES Clang)
            set_source_files_properties(src/crypto/cn/gpu/cn_gpu_arm.cpp PROPERTIES COMPILE_FLAGS "-O3")
        endif()
    else()
        list(APPEND SOURCES_CRYPTO
            src/crypto/cn/gpu/cn_gpu_avx.cpp
            src/crypto/cn/gpu/cn_gpu_ssse3.cpp
        )

        if (CMAKE_CXX_COMPILER_ID MATCHES GNU OR CMAKE_CXX_COMPILER_ID MATCHES Clang)
            set_source_files_properties(src/crypto/cn/gpu/cn_gpu_avx.cpp PROPERTIES COMPILE_FLAGS "-O3 -mavx2")
            set_source_files_properties(src/crypto/cn/gpu/cn_gpu_ssse3.cpp PROPERTIES COMPILE_FLAGS "-O3")
        elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
            set_source_files_properties(src/crypto/cn/gpu/cn_gpu_avx.cpp PROPERTIES COMPILE_FLAGS "-O3 -mavx2")
            set_source_files_properties(src/crypto/cn/gpu/cn_gpu_ssse3.cpp PROPERTIES COMPILE_FLAGS "-O1")
        elseif (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
            set_source_files_properties(src/crypto/cn/gpu/cn_gpu_avx.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX")
        endif()
    endif()

else()
    remove_definitions(/DXMRIG_ALGO_CN_GPU)
endif()
