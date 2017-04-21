set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes -mbmi2")
include_directories(../../..)
add_library(cryptonight_av2_aesni_bmi2 STATIC ../../../algo/cryptonight/cryptonight_av2_aesni_bmi2.c)
