if (WITH_ASM AND NOT XMRIG_ARM AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(XMRIG_ASM_LIBRARY "xmrig-asm")

    if (CMAKE_C_COMPILER_ID MATCHES MSVC)
        enable_language(ASM_MASM)

        if (MSVC_TOOLSET_VERSION GREATER_EQUAL 141)
            set(XMRIG_ASM_FILE "src/crypto/asm/cnv2_main_loop.asm")
        else()
            set(XMRIG_ASM_FILE "src/crypto/asm/win64/cnv2_main_loop.asm")
        endif()

        set_property(SOURCE ${XMRIG_ASM_FILE} PROPERTY ASM_MASM)
    else()
        enable_language(ASM)

        if (WIN32 AND CMAKE_C_COMPILER_ID MATCHES GNU)
            set(XMRIG_ASM_FILE "src/crypto/asm/win64/cnv2_main_loop.S")
        else()
            set(XMRIG_ASM_FILE "src/crypto/asm/cnv2_main_loop.S")
        endif()

        set_property(SOURCE ${XMRIG_ASM_FILE} PROPERTY C)
    endif()

    add_library(${XMRIG_ASM_LIBRARY} STATIC ${XMRIG_ASM_FILE})
    set(XMRIG_ASM_SOURCES src/crypto/Asm.h src/crypto/Asm.cpp)
    set_property(TARGET ${XMRIG_ASM_LIBRARY} PROPERTY LINKER_LANGUAGE C)
else()
    set(XMRIG_ASM_SOURCES "")
    set(XMRIG_ASM_LIBRARY "")
    add_definitions(/DXMRIG_NO_ASM)
endif()
