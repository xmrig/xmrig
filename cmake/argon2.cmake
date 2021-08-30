if (WITH_ARGON2)
    list(APPEND HEADERS
        src/crypto/argon2/Hash.h
        src/crypto/argon2/Impl.h
    )

    list(APPEND SOURCES
        src/crypto/argon2/Impl.cpp
    )

    add_subdirectory(src/3rdparty/argon2)
    list(APPEND LIBS argon2)
endif()
