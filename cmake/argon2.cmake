if (WITH_ARGON2)
    add_definitions(/Dxmlcore_ALGO_ARGON2)

    list(APPEND HEADERS_CRYPTO
        src/crypto/argon2/Hash.h
        src/crypto/argon2/Impl.h
    )

    list(APPEND SOURCES_CRYPTO
        src/crypto/argon2/Impl.cpp
    )

    add_subdirectory(src/3rdparty/argon2)
    set(ARGON2_LIBRARY argon2)
else()
    remove_definitions(/Dxmlcore_ALGO_ARGON2)
    set(ARGON2_LIBRARY "")
endif()
