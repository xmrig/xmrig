if (NOT CMAKE_SYSTEM_PROCESSOR)
    message(WARNING "CMAKE_SYSTEM_PROCESSOR not defined")
endif()


if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    add_definitions(/DRAPIDJSON_SSE2)
endif()


if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64)$")
    set(XMRIG_ARM     ON)
    set(XMRIG_ARMv8   ON)
    set(WITH_LIBCPUID OFF)

    add_definitions(/DXMRIG_ARM)
    add_definitions(/DXMRIG_ARMv8)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|armv7f|armv7s|armv7k|armv7-a|armv7l)$")
    set(XMRIG_ARM     ON)
    set(XMRIG_ARMv7   ON)
    set(WITH_LIBCPUID OFF)

    add_definitions(/DXMRIG_ARM)
    add_definitions(/DXMRIG_ARMv7)
endif()
