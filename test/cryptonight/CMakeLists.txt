set(SOURCES
    cryptonight.c
    ../../options.h
    ../../algo/cryptonight/cryptonight.h
    ../../algo/cryptonight/cryptonight.c
    ../../algo/cryptonight/cryptonight_av1_aesni.c
    ../../algo/cryptonight/cryptonight_av2_aesni_double.c
    ../../algo/cryptonight/cryptonight_av3_softaes.c
    ../../algo/cryptonight/cryptonight_av4_softaes_double.c
    ../../crypto/c_keccak.c
    ../../crypto/c_blake256.c
    ../../crypto/c_groestl.c
    ../../crypto/c_jh.c
    ../../crypto/c_skein.c
    ../../crypto/soft_aes.c
   )

add_executable(cryptonight_app ${SOURCES})
target_link_libraries(cryptonight_app unity)

include_directories(../..)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes -fno-strict-aliasing")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
add_definitions(-DBUILD_TEST)
add_definitions(-DXMRIG_NO_AEON)

add_test(cryptonight_test cryptonight_app)
