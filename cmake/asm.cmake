message("Generating ASM files")

# CN v1 original
set(ALGO "original")
set(ITERATIONS "524288") #0x80000
set(MASK "2097136") #0x1FFFF0

configure_file("src/crypto/asm/cnv1_main_loop_sandybridge.inc.in" "src/crypto/asm/cnv1_main_loop_sandybridge.inc")
configure_file("src/crypto/asm/cnv1_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/cnv1_main_loop_soft_aes_sandybridge.inc")

configure_file("src/crypto/asm/win/cnv1_main_loop_sandybridge.inc.in" "src/crypto/asm/win/cnv1_main_loop_sandybridge.inc")
configure_file("src/crypto/asm/win/cnv1_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/win/cnv1_main_loop_soft_aes_sandybridge.inc")

# CN v2 ORIGINAL
set(ALGO "originalv2")
set(ITERATIONS "524288") #0x80000
set(MASK "2097136") #0x1FFFF0

configure_file("src/crypto/asm/cnv2_main_loop_ivybridge.inc.in" "src/crypto/asm/cnv2_main_loop_ivybridge.inc")
configure_file("src/crypto/asm/cnv2_main_loop_bulldozer.inc.in" "src/crypto/asm/cnv2_main_loop_bulldozer.inc")
configure_file("src/crypto/asm/cnv2_main_loop_ryzen.inc.in" "src/crypto/asm/cnv2_main_loop_ryzen.inc")
configure_file("src/crypto/asm/cnv2_double_main_loop_sandybridge.inc.in" "src/crypto/asm/cnv2_double_main_loop_sandybridge.inc")
configure_file("src/crypto/asm/cnv2_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/cnv2_main_loop_soft_aes_sandybridge.inc")

configure_file("src/crypto/asm/win/cnv2_main_loop_ivybridge.inc.in" "src/crypto/asm/win/cnv2_main_loop_ivybridge.inc")
configure_file("src/crypto/asm/win/cnv2_main_loop_bulldozer.inc.in" "src/crypto/asm/win/cnv2_main_loop_bulldozer.inc")
configure_file("src/crypto/asm/win/cnv2_main_loop_ryzen.inc.in" "src/crypto/asm/win/cnv2_main_loop_ryzen.inc")
configure_file("src/crypto/asm/win/cnv2_double_main_loop_sandybridge.inc.in" "src/crypto/asm/win/cnv2_double_main_loop_sandybridge.inc")
configure_file("src/crypto/asm/win/cnv2_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/win/cnv2_main_loop_soft_aes_sandybridge.inc")

# CN v1 FAST
set(ALGO "fast")
set(ITERATIONS "262144") #0x40000
set(MASK "2097136") #0x1FFFF0

configure_file("src/crypto/asm/cnv1_main_loop_sandybridge.inc.in" "src/crypto/asm/cnv1_main_loop_fast_sandybridge.inc")
configure_file("src/crypto/asm/cnv1_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/cnv1_main_loop_fast_soft_aes_sandybridge.inc")

configure_file("src/crypto/asm/win/cnv1_main_loop_sandybridge.inc.in" "src/crypto/asm/win/cnv1_main_loop_fast_sandybridge.inc")
configure_file("src/crypto/asm/win/cnv1_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/win/cnv1_main_loop_fast_soft_aes_sandybridge.inc")

# CN v2 FAST
set(ALGO "fastv2")
set(ITERATIONS "262144") #0x40000
set(MASK "2097136") #0x1FFFF0

configure_file("src/crypto/asm/cnv2_main_loop_ivybridge.inc.in" "src/crypto/asm/cnv2_main_loop_fastv2_ivybridge.inc")
configure_file("src/crypto/asm/cnv2_main_loop_bulldozer.inc.in" "src/crypto/asm/cnv2_main_loop_fastv2_bulldozer.inc")
configure_file("src/crypto/asm/cnv2_main_loop_ryzen.inc.in" "src/crypto/asm/cnv2_main_loop_fastv2_ryzen.inc")
configure_file("src/crypto/asm/cnv2_double_main_loop_sandybridge.inc.in" "src/crypto/asm/cnv2_double_main_loop_fastv2_sandybridge.inc")
configure_file("src/crypto/asm/cnv2_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/cnv2_main_loop_fastv2_soft_aes_sandybridge.inc")

configure_file("src/crypto/asm/win/cnv2_main_loop_ivybridge.inc.in" "src/crypto/asm/win/cnv2_main_loop_fastv2_ivybridge.inc")
configure_file("src/crypto/asm/win/cnv2_main_loop_bulldozer.inc.in" "src/crypto/asm/win/cnv2_main_loop_fastv2_bulldozer.inc")
configure_file("src/crypto/asm/win/cnv2_main_loop_ryzen.inc.in" "src/crypto/asm/win/cnv2_main_loop_fastv2_ryzen.inc")
configure_file("src/crypto/asm/win/cnv2_double_main_loop_sandybridge.inc.in" "src/crypto/asm/win/cnv2_double_main_loop_fastv2_sandybridge.inc")
configure_file("src/crypto/asm/win/cnv2_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/win/cnv2_main_loop_fastv2_soft_aes_sandybridge.inc")

# CN LITE

set(ALGO "lite")
set(ITERATIONS "262144") #0x40000
set(MASK "1048560") #0xFFFF0

configure_file("src/crypto/asm/cnv1_main_loop_sandybridge.inc.in" "src/crypto/asm/cnv1_main_loop_lite_sandybridge.inc")
configure_file("src/crypto/asm/cnv1_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/cnv1_main_loop_lite_soft_aes_sandybridge.inc")

configure_file("src/crypto/asm/win/cnv1_main_loop_sandybridge.inc.in" "src/crypto/asm/win/cnv1_main_loop_lite_sandybridge.inc")
configure_file("src/crypto/asm/win/cnv1_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/win/cnv1_main_loop_lite_soft_aes_sandybridge.inc")

# CN UPX

set(ALGO "upx")
set(ITERATIONS "131072") #0x20000
set(MASK "1048560") #0xFFFF0

configure_file("src/crypto/asm/cnv1_main_loop_sandybridge.inc.in" "src/crypto/asm/cnv1_main_loop_upx_sandybridge.inc")
configure_file("src/crypto/asm/cnv1_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/cnv1_main_loop_upx_soft_aes_sandybridge.inc")

configure_file("src/crypto/asm/win/cnv1_main_loop_sandybridge.inc.in" "src/crypto/asm/win/cnv1_main_loop_upx_sandybridge.inc")
configure_file("src/crypto/asm/win/cnv1_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/win/cnv1_main_loop_upx_soft_aes_sandybridge.inc")

# CN V2 ULTRALITE
set(ALGO "ultralite")
set(ITERATIONS "65536") #0x10000
set(MASK "131056") #0x1FFF0

configure_file("src/crypto/asm/cnv2_main_loop_ivybridge.inc.in" "src/crypto/asm/cnv2_main_loop_ultralite_ivybridge.inc")
configure_file("src/crypto/asm/cnv2_main_loop_bulldozer.inc.in" "src/crypto/asm/cnv2_main_loop_ultralite_bulldozer.inc")
configure_file("src/crypto/asm/cnv2_main_loop_ryzen.inc.in" "src/crypto/asm/cnv2_main_loop_ultralite_ryzen.inc")
configure_file("src/crypto/asm/cnv2_double_main_loop_sandybridge.inc.in" "src/crypto/asm/cnv2_double_main_loop_ultralite_sandybridge.inc")
configure_file("src/crypto/asm/cnv2_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/cnv2_main_loop_ultralite_soft_aes_sandybridge.inc")

configure_file("src/crypto/asm/win/cnv2_main_loop_ivybridge.inc.in" "src/crypto/asm/win/cnv2_main_loop_ultralite_ivybridge.inc")
configure_file("src/crypto/asm/win/cnv2_main_loop_bulldozer.inc.in" "src/crypto/asm/win/cnv2_main_loop_ultralite_bulldozer.inc")
configure_file("src/crypto/asm/win/cnv2_main_loop_ryzen.inc.in" "src/crypto/asm/win/cnv2_main_loop_ultralite_ryzen.inc")
configure_file("src/crypto/asm/win/cnv2_double_main_loop_sandybridge.inc.in" "src/crypto/asm/win/cnv2_double_main_loop_ultralite_sandybridge.inc")
configure_file("src/crypto/asm/win/cnv2_main_loop_soft_aes_sandybridge.inc.in" "src/crypto/asm/win/cnv2_main_loop_ultralite_soft_aes_sandybridge.inc")

if (CMAKE_C_COMPILER_ID MATCHES MSVC)
    enable_language(ASM_MASM)
    set(XMRIG_ASM_FILE "src/crypto/asm/win/cn_main_loop.asm")
    set_property(SOURCE ${XMRIG_ASM_FILE} PROPERTY ASM_MASM)
    include_directories(${CMAKE_BINARY_DIR}/src/crypto/asm/win)
else()
    enable_language(ASM)

    if (WIN32 AND CMAKE_C_COMPILER_ID MATCHES GNU)
        set(XMRIG_ASM_FILE "src/crypto/asm/win/cn_main_loop_win_gcc.S")
    else()
        set(XMRIG_ASM_FILE "src/crypto/asm/cn_main_loop.S")
    endif()

    set_property(SOURCE ${XMRIG_ASM_FILE} PROPERTY C)
    include_directories(${CMAKE_BINARY_DIR}/src/crypto/asm/)
endif()

add_library(xmrig_asm STATIC ${XMRIG_ASM_FILE})
set_property(TARGET xmrig_asm PROPERTY LINKER_LANGUAGE C)