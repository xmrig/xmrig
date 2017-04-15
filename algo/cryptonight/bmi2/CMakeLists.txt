set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mbmi2")
add_library(cryptonight_av3_aesni_bmi2 STATIC ../cryptonight_av3_aesni_bmi2.c)
