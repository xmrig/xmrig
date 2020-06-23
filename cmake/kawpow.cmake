if (WITH_KAWPOW)
    add_definitions(/DXMRIG_ALGO_KAWPOW)

    list(APPEND HEADERS_CRYPTO
        src/crypto/kawpow/KPCache.h
        src/crypto/kawpow/KPHash.h
    )

    list(APPEND SOURCES_CRYPTO
        src/crypto/kawpow/KPCache.cpp
        src/crypto/kawpow/KPHash.cpp
    )

    add_subdirectory(src/3rdparty/libethash)
    set(ETHASH_LIBRARY ethash)
else()
    remove_definitions(/DXMRIG_ALGO_KAWPOW)
    set(ETHASH_LIBRARY "")
endif()
