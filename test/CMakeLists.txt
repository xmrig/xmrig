project("xmrig-test" C)
cmake_minimum_required(VERSION 3.0)

include(CTest)

add_subdirectory(unity)
add_subdirectory(cryptonight)
add_subdirectory(cryptonight_lite)
add_subdirectory(autoconf)