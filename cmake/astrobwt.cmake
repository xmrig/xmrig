if (WITH_ASTROBWT)
    add_definitions(/DXMRIG_ALGO_ASTROBWT)

    list(APPEND HEADERS_CRYPTO
        src/crypto/astrobwt/AstroBWT.h
    )

    list(APPEND SOURCES_CRYPTO
        src/crypto/astrobwt/AstroBWT.cpp
    )

    if (XMRIG_ARM)
        list(APPEND HEADERS_CRYPTO
            src/crypto/astrobwt/salsa20_ref/ecrypt-config.h
            src/crypto/astrobwt/salsa20_ref/ecrypt-machine.h
            src/crypto/astrobwt/salsa20_ref/ecrypt-portable.h
            src/crypto/astrobwt/salsa20_ref/ecrypt-sync.h
        )

        list(APPEND SOURCES_CRYPTO
            src/crypto/astrobwt/salsa20_ref/salsa20.c
        )
    else()
        if (CMAKE_SIZEOF_VOID_P EQUAL 8)
            add_definitions(/DASTROBWT_AVX2)
            if (CMAKE_C_COMPILER_ID MATCHES MSVC)
                enable_language(ASM_MASM)
                list(APPEND SOURCES_CRYPTO src/crypto/astrobwt/sha3_256_avx2.asm)
            else()
                enable_language(ASM)
                list(APPEND SOURCES_CRYPTO src/crypto/astrobwt/sha3_256_avx2.S)
            endif()
        endif()

        list(APPEND HEADERS_CRYPTO
            src/crypto/astrobwt/Salsa20.hpp
        )

        list(APPEND SOURCES_CRYPTO
            src/crypto/astrobwt/Salsa20.cpp
        )
    endif()
else()
    remove_definitions(/DXMRIG_ALGO_ASTROBWT)
endif()
