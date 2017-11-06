if (NOT CMAKE_SYSTEM_PROCESSOR)
    message(WARNING "CMAKE_SYSTEM_PROCESSOR not defined")
endif()


if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    add_definitions(/DRAPIDJSON_SSE2)
endif()


if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64)$")
    set(XMRIG_ARM64 ON)
    set(WITH_LIBCPUID OFF)

    add_definitions(/DXMRIG_ARM)
endif()
