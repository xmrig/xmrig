if (WITH_TLS)
    set(OPENSSL_ROOT_DIR ${XMRIG_DEPS})

    if (WIN32)
        set(OPENSSL_USE_STATIC_LIBS TRUE)
        set(OPENSSL_MSVC_STATIC_RT TRUE)
    elseif (APPLE)
        set(OPENSSL_USE_STATIC_LIBS TRUE)
    endif()

    if (BUILD_STATIC)
        set(OPENSSL_USE_STATIC_LIBS TRUE)
    endif()

    find_package(OpenSSL)

    if (NOT OPENSSL_FOUND)
        message(FATAL_ERROR "OpenSSL NOT found: use `-DWITH_TLS=OFF` to build without TLS support")
    endif()

    add_definitions(-DXMRIG_FEATURE_TLS)

    message(STATUS "WITH_TLS        \t= ON (v${OPENSSL_VERSION})")
else()
    message(STATUS "WITH_TLS        \t= ${WITH_TLS}")
endif()
