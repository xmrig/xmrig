if (WITH_KAWPOW)
    list(APPEND HEADERS
        src/crypto/kawpow/KPCache.h
        src/crypto/kawpow/KPHash.h
    )

    list(APPEND SOURCES
        src/crypto/kawpow/KPCache.cpp
        src/crypto/kawpow/KPHash.cpp
    )

    add_subdirectory(src/base/3rdparty/libethash)
    list(APPEND LIBS ethash)
endif()
