cmake_minimum_required(VERSION 3.5)

project(argon2 C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(CheckCSourceCompiles)

set(ARGON2_SOURCES
    lib/argon2.c
    lib/core.c
    lib/encoding.c
    lib/genkat.c
    lib/impl-select.c
    lib/blake2/blake2.c
    )

set(ARGON2_X86_64_ENABLED ON)
set(ARGON2_X86_64_LIBS    argon2-sse2 argon2-ssse3 argon2-xop argon2-avx2 argon2-avx512f)
set(ARGON2_X86_64_SOURCES arch/x86_64/lib/argon2-arch.c)

if (CMAKE_C_COMPILER_ID MATCHES MSVC)
    function(add_feature_impl FEATURE MSVC_FLAG DEF)
        add_library(argon2-${FEATURE} STATIC arch/x86_64/lib/argon2-${FEATURE}.c)
        target_include_directories(argon2-${FEATURE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
        target_include_directories(argon2-${FEATURE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
        set_target_properties(argon2-${FEATURE} PROPERTIES POSITION_INDEPENDENT_CODE True)

        target_compile_options(argon2-${FEATURE} PRIVATE ${MSVC_FLAG})
        target_compile_definitions(argon2-${FEATURE} PRIVATE ${DEF})
    endfunction()

    add_feature_impl(sse2    ""              HAVE_SSE2)
    add_feature_impl(ssse3   "/arch:SSSE3"   HAVE_SSSE3)
    add_feature_impl(xop     ""              HAVE_XOP)
    add_feature_impl(avx2    "/arch:AVX2"    HAVE_AVX2)
    add_feature_impl(avx512f "/arch:AVX512F" HAVE_AVX512F)
elseif (NOT XMRIG_ARM AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    function(add_feature_impl FEATURE GCC_FLAG DEF)
        add_library(argon2-${FEATURE} STATIC arch/x86_64/lib/argon2-${FEATURE}.c)
        target_include_directories(argon2-${FEATURE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
        target_include_directories(argon2-${FEATURE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
        set_target_properties(argon2-${FEATURE} PROPERTIES POSITION_INDEPENDENT_CODE True)

        message("-- argon2: detecting feature '${FEATURE}'...")
        file(READ arch/x86_64/src/test-feature-${FEATURE}.c SOURCE_CODE)

        # try without flag:
        check_c_source_compiles("${SOURCE_CODE}" FEATURE_${FEATURE}_NOFLAG)
        set(HAS_FEATURE ${FEATURE_${FEATURE}_NOFLAG})
        if (NOT "${HAS_FEATURE}")
            # try with -m<feature> flag:
            set(CMAKE_REQUIRED_FLAGS ${GCC_FLAG})
            check_c_source_compiles("${SOURCE_CODE}" FEATURE_${FEATURE}_FLAG)
            set(CMAKE_REQUIRED_FLAGS "")

            set(HAS_FEATURE ${FEATURE_${FEATURE}_FLAG})
            if (${HAS_FEATURE})
                target_compile_options(argon2-${FEATURE} PRIVATE ${GCC_FLAG})
            endif()
        endif()

        if (${HAS_FEATURE})
            message("-- argon2: feature '${FEATURE}' detected!")
            target_compile_definitions(argon2-${FEATURE} PRIVATE ${DEF})
        endif()
    endfunction()

    add_feature_impl(sse2    -msse2    HAVE_SSE2)
    add_feature_impl(ssse3   -mssse3   HAVE_SSSE3)
    add_feature_impl(xop     -mxop     HAVE_XOP)
    add_feature_impl(avx2    -mavx2    HAVE_AVX2)
    add_feature_impl(avx512f -mavx512f HAVE_AVX512F)
else()
    set(ARGON2_X86_64_ENABLED OFF)
    list(APPEND ARGON2_SOURCES arch/generic/lib/argon2-arch.c)
endif()

if (ARGON2_X86_64_ENABLED)
    set(ARGON2_LIBS ${ARGON2_X86_64_LIBS})
    list(APPEND ARGON2_SOURCES ${ARGON2_X86_64_SOURCES})
endif()

add_library(argon2 STATIC ${ARGON2_SOURCES})
target_link_libraries(argon2 ${ARGON2_LIBS})

target_include_directories(argon2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_include_directories(argon2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
