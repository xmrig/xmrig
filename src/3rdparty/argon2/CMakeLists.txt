cmake_minimum_required(VERSION 2.6)

project(Argon2 C)
set(ARGON2_VERSION 1.0)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(CheckCSourceCompiles)
find_package(Threads REQUIRED)

add_library(argon2-interface INTERFACE)
target_include_directories(argon2-interface INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_library(argon2-internal INTERFACE)
target_include_directories(argon2-internal INTERFACE lib lib/blake2)
target_link_libraries(argon2-internal INTERFACE argon2-interface)

add_library(argon2 STATIC
    lib/argon2.c
    lib/core.c
    lib/encoding.c
    lib/genkat.c
    lib/impl-select.c
    lib/thread.c
    lib/blake2/blake2.c
)
target_link_libraries(argon2
    PUBLIC argon2-interface ${CMAKE_THREAD_LIBS_INIT}
    PRIVATE argon2-internal
)

set_property(TARGET argon2 PROPERTY VERSION ${Upstream_VERSION})
set_property(TARGET argon2 PROPERTY SOVERSION 1)
set_property(TARGET argon2 PROPERTY INTERFACE_ARGON2_MAJOR_VERSION 1)
set_property(TARGET argon2 APPEND PROPERTY
    COMPATIBLE_INTERFACE_STRING ARGON2_MAJOR_VERSION
)

if (CMAKE_C_COMPILER_ID MATCHES MSVC)
    function(add_feature_impl FEATURE MSVC_FLAG DEF)
        add_library(argon2-${FEATURE} STATIC arch/x86_64/lib/argon2-${FEATURE}.c)
        target_link_libraries(argon2-${FEATURE} PRIVATE argon2-internal)
        set_target_properties(argon2-${FEATURE} PROPERTIES POSITION_INDEPENDENT_CODE True)

        target_compile_options(argon2-${FEATURE} PRIVATE ${MSVC_FLAG})
        target_compile_definitions(argon2-${FEATURE} PRIVATE ${DEF})
        target_link_libraries(argon2 PUBLIC argon2-${FEATURE})
    endfunction()

    add_feature_impl(sse2    ""              HAVE_SSE2)
    add_feature_impl(ssse3   ""/arch:SSSE3"" HAVE_SSSE3)
    add_feature_impl(xop     ""              HAVE_XOP)
    add_feature_impl(avx2    "/arch:AVX2"    HAVE_AVX2)
    add_feature_impl(avx512f "/arch:AVX512F" HAVE_AVX512F)

    target_sources(argon2 PRIVATE arch/x86_64/lib/argon2-arch.c arch/x86_64/lib/cpu-flags.c)
elseif (NOT XMRIG_ARM AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    function(add_feature_impl FEATURE GCC_FLAG DEF)
        add_library(argon2-${FEATURE} STATIC arch/x86_64/lib/argon2-${FEATURE}.c)
        target_link_libraries(argon2-${FEATURE} PRIVATE argon2-internal)
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

        target_link_libraries(argon2 PUBLIC argon2-${FEATURE})
    endfunction()

    target_include_directories(argon2-internal INTERFACE arch/x86_64/lib)

    add_feature_impl(sse2    -msse2    HAVE_SSE2)
    add_feature_impl(ssse3   -mssse3   HAVE_SSSE3)
    add_feature_impl(xop     -mxop     HAVE_XOP)
    add_feature_impl(avx2    -mavx2    HAVE_AVX2)
    add_feature_impl(avx512f -mavx512f HAVE_AVX512F)

    target_sources(argon2 PRIVATE arch/x86_64/lib/argon2-arch.c arch/x86_64/lib/cpu-flags.c)
else()
    target_sources(argon2 PRIVATE arch/generic/lib/argon2-arch.c)
endif()
