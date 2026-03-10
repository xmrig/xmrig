if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(XMRIG_64_BIT ON)
    add_definitions(-DXMRIG_64_BIT)
else()
    set(XMRIG_64_BIT OFF)
endif()

if (NOT CMAKE_SYSTEM_PROCESSOR)
    message(WARNING "CMAKE_SYSTEM_PROCESSOR not defined")
endif()

include(CheckCXXCompilerFlag)

if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    set(VAES_SUPPORTED ON)
else()
    CHECK_CXX_COMPILER_FLAG("-mavx2 -mvaes" VAES_SUPPORTED)
endif()

if (NOT VAES_SUPPORTED)
    set(WITH_VAES OFF)
endif()

# Detect RISC-V architecture early (before it's used below)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv64|riscv|rv64)$")
    set(RISCV_TARGET 64)
    set(XMRIG_RISCV ON)
    add_definitions(-DXMRIG_RISCV)
    message(STATUS "Detected RISC-V 64-bit architecture (${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv32|rv32)$")
    set(RISCV_TARGET 32)
    set(XMRIG_RISCV ON)
    add_definitions(-DXMRIG_RISCV)
    message(STATUS "Detected RISC-V 32-bit architecture (${CMAKE_SYSTEM_PROCESSOR})")
endif()

if (XMRIG_64_BIT AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    add_definitions(-DRAPIDJSON_SSE2)
else()
    set(WITH_SSE4_1 OFF)
    set(WITH_AVX2 OFF)
    set(WITH_VAES OFF)
endif()

# Disable x86-specific features for RISC-V
if (XMRIG_RISCV)
    set(WITH_SSE4_1 OFF)
    set(WITH_AVX2 OFF)
    set(WITH_VAES OFF)

    # default build uses the RV64GC baseline
    set(RVARCH "rv64gc")

    # for native builds, enable Zba and Zbb if supported by the CPU
    if(ARCH STREQUAL "native")
        enable_language(ASM)

        try_run(RANDOMX_VECTOR_RUN_FAIL
            RANDOMX_VECTOR_COMPILE_OK
            ${CMAKE_CURRENT_BINARY_DIR}/
            ${CMAKE_CURRENT_SOURCE_DIR}/src/crypto/randomx/tests/riscv64_vector.s
            COMPILE_DEFINITIONS "-march=rv64gcv_zicbop")

        if (RANDOMX_VECTOR_COMPILE_OK AND NOT RANDOMX_VECTOR_RUN_FAIL)
            set(RVARCH "${RVARCH}v_zicbop")
            add_definitions(-DXMRIG_RVV_ENABLED)
            message(STATUS "RISC-V vector extension detected")
        endif()

        try_run(RANDOMX_ZBA_RUN_FAIL
            RANDOMX_ZBA_COMPILE_OK
            ${CMAKE_CURRENT_BINARY_DIR}/
            ${CMAKE_CURRENT_SOURCE_DIR}/src/crypto/randomx/tests/riscv64_zba.s
            COMPILE_DEFINITIONS "-march=rv64gc_zba")

        if (RANDOMX_ZBA_COMPILE_OK AND NOT RANDOMX_ZBA_RUN_FAIL)
            set(RVARCH "${RVARCH}_zba")
            message(STATUS "RISC-V zba extension detected")
        endif()

        try_run(RANDOMX_ZBB_RUN_FAIL
            RANDOMX_ZBB_COMPILE_OK
            ${CMAKE_CURRENT_BINARY_DIR}/
            ${CMAKE_CURRENT_SOURCE_DIR}/src/crypto/randomx/tests/riscv64_zbb.s
            COMPILE_DEFINITIONS "-march=rv64gc_zbb")

        if (RANDOMX_ZBB_COMPILE_OK AND NOT RANDOMX_ZBB_RUN_FAIL)
            set(RVARCH "${RVARCH}_zbb")
            message(STATUS "RISC-V zbb extension detected")
        endif()
    endif()

    message(STATUS "Using -march=${RVARCH}")
endif()

add_definitions(-DRAPIDJSON_WRITE_DEFAULT_FLAGS=6) # rapidjson::kWriteNanAndInfFlag | rapidjson::kWriteNanAndInfNullFlag

if (ARM_V8)
    set(ARM_TARGET 8)
elseif (ARM_V7)
    set(ARM_TARGET 7)
endif()

if (NOT ARM_TARGET)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|armv8-a)$")
        set(ARM_TARGET 8)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|armv7f|armv7s|armv7k|armv7-a|armv7l|armv7ve|armv8l)$")
        set(ARM_TARGET 7)
    endif()
endif()

if (ARM_TARGET AND ARM_TARGET GREATER 6)
    set(XMRIG_ARM ON)
    add_definitions(-DXMRIG_ARM=${ARM_TARGET})

    message(STATUS "Use ARM_TARGET=${ARM_TARGET} (${CMAKE_SYSTEM_PROCESSOR})")

    if (ARM_TARGET EQUAL 8 AND (CMAKE_CXX_COMPILER_ID MATCHES GNU OR CMAKE_CXX_COMPILER_ID MATCHES Clang))
        CHECK_CXX_COMPILER_FLAG(-march=armv8-a+crypto XMRIG_ARM_CRYPTO)

        if (XMRIG_ARM_CRYPTO)
            add_definitions(-DXMRIG_ARM_CRYPTO)
            set(ARM8_CXX_FLAGS "-march=armv8-a+crypto")
        else()
            set(ARM8_CXX_FLAGS "-march=armv8-a")
        endif()
    endif()
endif()

if (WITH_SSE4_1)
    add_definitions(-DXMRIG_FEATURE_SSE4_1)
endif()

if (WITH_AVX2)
    add_definitions(-DXMRIG_FEATURE_AVX2)
endif()
