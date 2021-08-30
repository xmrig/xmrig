if (WITH_RANDOMX AND NOT WITH_ARGON2)
    set(WITH_ARGON2 ON)
endif()

message(STATUS "WITH_CN_LITE\t= ${WITH_CN_LITE}")
message(STATUS "WITH_CN_HEAVY\t= ${WITH_CN_HEAVY}")
message(STATUS "WITH_CN_PICO\t= ${WITH_CN_PICO}")
message(STATUS "WITH_CN_FEMTO\t= ${WITH_CN_FEMTO}")
message(STATUS "WITH_RANDOMX\t= ${WITH_RANDOMX}")
message(STATUS "WITH_ARGON2\t= ${WITH_ARGON2}")
message(STATUS "WITH_ASTROBWT\t= ${WITH_ASTROBWT}")
message(STATUS "WITH_KAWPOW\t= ${WITH_KAWPOW}")

if (WITH_CN_LITE)
    add_definitions(-DXMRIG_ALGO_CN_LITE)
endif()

if (WITH_CN_HEAVY)
    add_definitions(-DXMRIG_ALGO_CN_HEAVY)
endif()

if (WITH_CN_PICO)
    add_definitions(-DXMRIG_ALGO_CN_PICO)
endif()

if (WITH_CN_FEMTO)
    add_definitions(-DXMRIG_ALGO_CN_FEMTO)
endif()

if (WITH_RANDOMX)
    add_definitions(-DXMRIG_ALGO_RANDOMX)
endif()

if (WITH_ARGON2)
    add_definitions(-DXMRIG_ALGO_ARGON2)
endif()

if (WITH_ASTROBWT)
    add_definitions(-DXMRIG_ALGO_ASTROBWT)
endif()

if (WITH_KAWPOW)
    add_definitions(-DXMRIG_ALGO_KAWPOW)
endif()
