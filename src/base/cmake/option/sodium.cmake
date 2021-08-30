message(STATUS "WITH_SODIUM     \t= ${WITH_SODIUM}")

if (WITH_SODIUM)
    find_package(Sodium REQUIRED)

    add_definitions(-DXMRIG_FEATURE_SODIUM)
endif()
