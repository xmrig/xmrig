if (WITH_GHOSTRIDER)
    add_subdirectory(src/crypto/ghostrider)
    list(APPEND LIBS ghostrider)
endif()
