set(SOURCES
    ../../algo/cryptonight/cryptonight.h
    ../../algo/cryptonight/cryptonight_common.c
    ../../algo/cryptonight/cryptonight_av4_legacy.c
    ../../crypto/c_keccak.c
    ../../crypto/c_blake256.c
    ../../crypto/c_groestl.c
    ../../crypto/c_jh.c
    ../../crypto/c_skein.c
    ../../crypto/oaes_config.h
    ../../crypto/oaes_lib.h
    ../../crypto/oaes_lib.c
    ../../crypto/aesb.c
   )

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    add_subdirectory(bmi2)

    add_executable(cryptonight_app ${SOURCES}
        cryptonight.c
        ../../algo/cryptonight/cryptonight_av1_aesni.c
        ../../algo/cryptonight/cryptonight_av2_aesni_wolf.c
        ../../algo/cryptonight/cryptonight_av5_aesni_experimental.c
    )

    target_link_libraries(cryptonight_app unity cryptonight_av3_aesni_bmi2)
else()
    add_executable(cryptonight_app ${SOURCES}
        cryptonight32.c
        ../../algo/cryptonight/cryptonight_av1_aesni32.c
    )

    target_link_libraries(cryptonight_app unity)
endif()



include_directories(../..)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes -fno-strict-aliasing")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
add_definitions(-DBUILD_TEST)

add_test(cryptonight_test cryptonight_app)
