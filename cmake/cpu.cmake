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

    enable_language(ASM)

    try_run(RANDOMX_VECTOR_RUN_FAIL
        RANDOMX_VECTOR_COMPILE_OK
        ${CMAKE_CURRENT_BINARY_DIR}/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/crypto/randomx/tests/riscv64_vector.s
        COMPILE_DEFINITIONS "-march=rv64gcv")

    if (RANDOMX_VECTOR_COMPILE_OK AND NOT RANDOMX_VECTOR_RUN_FAIL)
        set(RVARCH_V ON)
        message(STATUS "RISC-V vector extension detected")
    else()
        set(RVARCH_V OFF)
    endif()

    try_run(RANDOMX_ZICBOP_RUN_FAIL
        RANDOMX_ZICBOP_COMPILE_OK
        ${CMAKE_CURRENT_BINARY_DIR}/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/crypto/randomx/tests/riscv64_zicbop.s
        COMPILE_DEFINITIONS "-march=rv64gc_zicbop")

    if (RANDOMX_ZICBOP_COMPILE_OK AND NOT RANDOMX_ZICBOP_RUN_FAIL)
        set(RVARCH_ZICBOP ON)
        message(STATUS "RISC-V zicbop extension detected")
    else()
        set(RVARCH_ZICBOP OFF)
    endif()

    try_run(RANDOMX_ZBA_RUN_FAIL
        RANDOMX_ZBA_COMPILE_OK
        ${CMAKE_CURRENT_BINARY_DIR}/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/crypto/randomx/tests/riscv64_zba.s
        COMPILE_DEFINITIONS "-march=rv64gc_zba")

    if (RANDOMX_ZBA_COMPILE_OK AND NOT RANDOMX_ZBA_RUN_FAIL)
        set(RVARCH_ZBA ON)
        message(STATUS "RISC-V zba extension detected")
    else()
        set(RVARCH_ZBA OFF)
    endif()

    try_run(RANDOMX_ZBB_RUN_FAIL
        RANDOMX_ZBB_COMPILE_OK
        ${CMAKE_CURRENT_BINARY_DIR}/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/crypto/randomx/tests/riscv64_zbb.s
        COMPILE_DEFINITIONS "-march=rv64gc_zbb")

    if (RANDOMX_ZBB_COMPILE_OK AND NOT RANDOMX_ZBB_RUN_FAIL)
        set(RVARCH_ZBB ON)
        message(STATUS "RISC-V zbb extension detected")
    else()
        set(RVARCH_ZBB OFF)
    endif()

    try_run(RANDOMX_ZVKB_RUN_FAIL
        RANDOMX_ZVKB_COMPILE_OK
        ${CMAKE_CURRENT_BINARY_DIR}/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/crypto/randomx/tests/riscv64_zvkb.s
        COMPILE_DEFINITIONS "-march=rv64gcv_zvkb")

    if (RANDOMX_ZVKB_COMPILE_OK AND NOT RANDOMX_ZVKB_RUN_FAIL)
        set(RVARCH_ZVKB ON)
        message(STATUS "RISC-V zvkb extension detected")
    else()
        set(RVARCH_ZVKB OFF)
    endif()

    try_run(RANDOMX_ZVKNED_RUN_FAIL
        RANDOMX_ZVKNED_COMPILE_OK
        ${CMAKE_CURRENT_BINARY_DIR}/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/crypto/randomx/tests/riscv64_zvkned.s
        COMPILE_DEFINITIONS "-march=rv64gcv_zvkned")

    if (RANDOMX_ZVKNED_COMPILE_OK AND NOT RANDOMX_ZVKNED_RUN_FAIL)
        set(RVARCH_ZVKNED ON)
        message(STATUS "RISC-V zvkned extension detected")
    else()
        set(RVARCH_ZVKNED OFF)
    endif()

    # for native builds, enable Zba and Zbb if supported by the CPU
    if (ARCH STREQUAL "native")
        if (RVARCH_V)
            set(RVARCH "${RVARCH}v")
        endif()
        if (RVARCH_ZICBOP)
            set(RVARCH "${RVARCH}_zicbop")
        endif()
        if (RVARCH_ZBA)
            set(RVARCH "${RVARCH}_zba")
        endif()
        if (RVARCH_ZBB)
            set(RVARCH "${RVARCH}_zbb")
        endif()
        if (RVARCH_ZVKB)
            set(RVARCH "${RVARCH}_zvkb")
        endif()
        if (RVARCH_ZVKNED)
            set(RVARCH "${RVARCH}_zvkned")
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

    if (ARM_TARGET EQUAL 8 AND
        (CMAKE_CXX_COMPILER_ID MATCHES GNU OR CMAKE_CXX_COMPILER_ID MATCHES Clang))

        set(XMRIG_ARM_V9 OFF)
        set(XMRIG_ARM_V9_NAME "")

        if (NOT CMAKE_CROSSCOMPILING)
            if (APPLE)
                execute_process(
                    COMMAND /usr/sbin/sysctl -n machdep.cpu.brand_string
                    OUTPUT_VARIABLE XMRIG_ARM_BRAND
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )

                if (XMRIG_ARM_BRAND MATCHES "^Apple M4")
                    set(XMRIG_ARM_V9 ON)
                    set(XMRIG_ARM_V9_NAME "Apple M4")
                endif()
            elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
                file(GLOB XMRIG_ARM_MIDR_FILES
                    "/sys/devices/system/cpu/cpu[0-9]*/regs/identification/midr_el1"
                )

                set(XMRIG_HAS_CORTEX_A520 OFF)
                set(XMRIG_HAS_CORTEX_A720 OFF)

                foreach(XMRIG_MIDR_FILE IN LISTS XMRIG_ARM_MIDR_FILES)
                    if (EXISTS "${XMRIG_MIDR_FILE}")
                        file(READ "${XMRIG_MIDR_FILE}" XMRIG_MIDR)
                        string(STRIP "${XMRIG_MIDR}" XMRIG_MIDR)
                        string(TOLOWER "${XMRIG_MIDR}" XMRIG_MIDR)

                        if (XMRIG_MIDR MATCHES "410fd80[0-9a-f]$")
                            set(XMRIG_HAS_CORTEX_A520 ON)
                        elseif (XMRIG_MIDR MATCHES "410fd81[0-9a-f]$")
                            set(XMRIG_HAS_CORTEX_A720 ON)
                        endif()
                    endif()
                endforeach()

                if (XMRIG_HAS_CORTEX_A520 OR XMRIG_HAS_CORTEX_A720)
                    set(XMRIG_ARM_V9 ON)

                    if (XMRIG_HAS_CORTEX_A520 AND XMRIG_HAS_CORTEX_A720)
                        set(XMRIG_ARM_V9_NAME "Cortex-A520/A720")
                    elseif (XMRIG_HAS_CORTEX_A520)
                        set(XMRIG_ARM_V9_NAME "Cortex-A520")
                    else()
                        set(XMRIG_ARM_V9_NAME "Cortex-A720")
                    endif()
                endif()
            endif()
        endif()

        if (XMRIG_ARM_V9)
            CHECK_CXX_COMPILER_FLAG("-mcpu=native" XMRIG_ARM_NATIVE_SUPPORTED)

            if (XMRIG_ARM_NATIVE_SUPPORTED)
                if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
                    XMRIG_HAS_CORTEX_A520 AND
                    XMRIG_HAS_CORTEX_A720)
                    set(ARM8_CXX_FLAGS
                        "-march=armv9.2-a+crypto -mtune=cortex-a720"
                    )
                else()
                    set(ARM8_CXX_FLAGS "-mcpu=native")
                endif()

                add_definitions(-DXMRIG_ARM_V9)

                message(STATUS
                    "Detected ARMv9 CPU: ${XMRIG_ARM_V9_NAME}, using ${ARM8_CXX_FLAGS}"
                )
            else()
                message(WARNING
                    "Detected ARMv9 CPU, but compiler does not support -mcpu=native"
                )
            endif()
        endif()

        if (NOT ARM8_CXX_FLAGS)
            CHECK_CXX_COMPILER_FLAG(
                "-march=armv8-a+crypto"
                XMRIG_ARM_CRYPTO_FLAG
            )

            if (XMRIG_ARM_CRYPTO_FLAG)
                set(ARM8_CXX_FLAGS "-march=armv8-a+crypto")
            else()
                set(ARM8_CXX_FLAGS "-march=armv8-a")
            endif()
        endif()

        set(CMAKE_REQUIRED_FLAGS_SAVE "${CMAKE_REQUIRED_FLAGS}")
        set(CMAKE_REQUIRED_FLAGS
            "${CMAKE_REQUIRED_FLAGS} ${ARM8_CXX_FLAGS}"
        )

        include(CheckCXXSourceCompiles)

        check_cxx_source_compiles("
            #include <arm_neon.h>

            int main()
            {
                uint8x16_t a = vdupq_n_u8(0);
                uint8x16_t b = vdupq_n_u8(0);
                a = vaeseq_u8(a, b);

                return vgetq_lane_u8(a, 0);
            }
        " XMRIG_ARM_CRYPTO)

        set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_SAVE}")
        unset(CMAKE_REQUIRED_FLAGS_SAVE)

        if (XMRIG_ARM_CRYPTO)
            add_definitions(-DXMRIG_ARM_CRYPTO)
        endif()

        message(STATUS "ARM compiler flags: ${ARM8_CXX_FLAGS}")
    endif()
endif()

if (WITH_SSE4_1)
    add_definitions(-DXMRIG_FEATURE_SSE4_1)
endif()

if (WITH_AVX2)
    add_definitions(-DXMRIG_FEATURE_AVX2)
endif()
