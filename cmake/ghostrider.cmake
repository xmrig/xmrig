if (WITH_GHOSTRIDER OR WITH_SHA256CSM) # WITH_SHA256CSM for sph_sha2
    add_definitions(/DXMRIG_ALGO_GHOSTRIDER)
    add_subdirectory(src/crypto/ghostrider)
    set(GHOSTRIDER_LIBRARY ghostrider)
else()
    remove_definitions(/DXMRIG_ALGO_GHOSTRIDER)
    set(GHOSTRIDER_LIBRARY "")
endif()
