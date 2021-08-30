if (WITH_ASTROBWT)
    list(APPEND HEADERS src/crypto/astrobwt/AstroBWT.h)
    list(APPEND SOURCES src/crypto/astrobwt/AstroBWT.cpp)

    if (XMRIG_ARM)
        list(APPEND HEADERS
            src/crypto/astrobwt/salsa20_ref/ecrypt-config.h
            src/crypto/astrobwt/salsa20_ref/ecrypt-machine.h
            src/crypto/astrobwt/salsa20_ref/ecrypt-portable.h
            src/crypto/astrobwt/salsa20_ref/ecrypt-sync.h
        )

        list(APPEND SOURCES
            src/crypto/astrobwt/salsa20_ref/salsa20.c
        )
    else()
        if (CMAKE_SIZEOF_VOID_P EQUAL 8)
            add_definitions(/DASTROBWT_AVX2)
            list(APPEND SOURCES src/crypto/astrobwt/xmm6int/salsa20_xmm6int-avx2.c)

            if (CMAKE_C_COMPILER_ID MATCHES GNU OR CMAKE_C_COMPILER_ID MATCHES Clang)
                set_source_files_properties(src/crypto/astrobwt/xmm6int/salsa20_xmm6int-avx2.c PROPERTIES COMPILE_FLAGS -mavx2)
            endif()

            if (CMAKE_C_COMPILER_ID MATCHES MSVC)
                enable_language(ASM_MASM)
                list(APPEND SOURCES src/crypto/astrobwt/sha3_256_avx2.asm)
            else()
                enable_language(ASM)
                list(APPEND SOURCES src/crypto/astrobwt/sha3_256_avx2.S)
            endif()
        endif()

        list(APPEND HEADERS src/crypto/astrobwt/Salsa20.hpp)
        list(APPEND SOURCES src/crypto/astrobwt/Salsa20.cpp)
    endif()
endif()
