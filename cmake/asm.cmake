if (WITH_ASM AND NOT XMRIG_ARM)
    set(XMRIG_ASM_LIBRARY "xmrig-asm")

    if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        enable_language(ASM_MASM)
        set_property(SOURCE "src/crypto/asm/cnv2_main_loop.asm" PROPERTY ASM_MASM)
        add_library(${XMRIG_ASM_LIBRARY} STATIC
            "src/crypto/asm/cnv2_main_loop.asm"
            )
    else()
        enable_language(ASM)
        set_property(SOURCE "src/crypto/asm/cnv2_main_loop.S" PROPERTY C)
        add_library(${XMRIG_ASM_LIBRARY} STATIC
            "src/crypto/asm/cnv2_main_loop.S"
            )
    endif()

    set_property(TARGET ${XMRIG_ASM_LIBRARY} PROPERTY LINKER_LANGUAGE C)
else()
#    set(XMRIG_ASM_SOURCES "")
    set(XMRIG_ASM_LIBRARY "")
    add_definitions(/DXMRIG_NO_ASM)
endif()
