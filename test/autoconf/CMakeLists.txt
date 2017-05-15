set(SOURCES
    autoconf.c
    ../../cpu.h
    ../../cpu.c
   )

add_executable(autoconf_app ${SOURCES})
target_link_libraries(autoconf_app unity)

include_directories(../..)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-strict-aliasing")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
add_definitions(-DBUILD_TEST)

add_test(autoconf_test autoconf_app)
