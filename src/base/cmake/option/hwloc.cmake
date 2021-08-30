message(STATUS "WITH_HWLOC      \t= ${WITH_HWLOC} (DEBUG=${HWLOC_DEBUG})")

if (WITH_HWLOC)
    if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        add_subdirectory(src/base/3rdparty/hwloc)
        set(HWLOC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/base/3rdparty/hwloc/include)
        set(HWLOC_LIBRARY hwloc)
    else()
        find_package(HWLOC REQUIRED)
    endif()

    add_definitions(-DXMRIG_FEATURE_HWLOC)

    if (HWLOC_DEBUG)
        add_definitions(-DXMRIG_HWLOC_DEBUG)
    endif()
endif()
