if (WITH_VERUSHASH)
    add_definitions(/DXMRIG_ALGO_VERUSHASH)
    set(VERUSHASH_LIBRARY verushash)

    list(APPEND HEADERS_CRYPTO
        src/crypto/verus/haraka.h
        src/crypto/verus/haraka_portable.h
        src/crypto/verus/verus_hash.h
    )

    list(APPEND SOURCES_CRYPTO
        src/crypto/verus/haraka.c
        src/crypto/verus/equi-stratum.c
        src/crypto/verus/haraka_portable.c
        src/crypto/verus/verus_hash.c
    )

else()
    remove_definitions(/DXMRIG_ALGO_VERUSHASH)
    set(VERUSHASH_LIBRARY "")
endif()
