if (WITH_DMI AND (xmlcore_OS_WIN OR xmlcore_OS_LINUX OR xmlcore_OS_FREEBSD OR (xmlcore_OS_MACOS AND NOT xmlcore_ARM)))
    set(WITH_DMI ON)
else()
    set(WITH_DMI OFF)
endif()

if (WITH_DMI)
    add_definitions(/Dxmlcore_FEATURE_DMI)

    list(APPEND HEADERS
        src/hw/dmi/DmiBoard.h
        src/hw/dmi/DmiMemory.h
        src/hw/dmi/DmiReader.h
        src/hw/dmi/DmiTools.h
        )

    list(APPEND SOURCES
        src/hw/dmi/DmiBoard.cpp
        src/hw/dmi/DmiMemory.cpp
        src/hw/dmi/DmiReader.cpp
        src/hw/dmi/DmiTools.cpp
        )

    if (xmlcore_OS_WIN)
        list(APPEND SOURCES src/hw/dmi/DmiReader_win.cpp)
    elseif(xmlcore_OS_LINUX OR xmlcore_OS_FREEBSD)
        list(APPEND SOURCES src/hw/dmi/DmiReader_unix.cpp)
    elseif(xmlcore_OS_MACOS)
        list(APPEND SOURCES src/hw/dmi/DmiReader_mac.cpp)
        find_library(CORESERVICES_LIBRARY CoreServices)
        list(APPEND EXTRA_LIBS ${CORESERVICES_LIBRARY})
    endif()
else()
    remove_definitions(/Dxmlcore_FEATURE_DMI)
endif()
