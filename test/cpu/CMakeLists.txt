set(SOURCES
    test_cpu.cpp
    ../../src/Cpu.cpp
        )

include_directories(../../src)

add_executable(test_cpu ${SOURCES})
target_link_libraries(test_cpu unity xmrig_os_dependencies ${EXTRA_LIBS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes -fno-strict-aliasing")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
add_definitions(-DBUILD_TEST)

add_test(test_cpu test_cpu)
