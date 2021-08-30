message(STATUS "WITH_HTTP\t= ${WITH_HTTP}")

if (WITH_HTTP)
    add_definitions(-DXMRIG_FEATURE_HTTP -DXMRIG_FEATURE_API)
endif()
