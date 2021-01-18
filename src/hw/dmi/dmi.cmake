if (WITH_DMI)
    add_definitions(/DXMRIG_FEATURE_DMI)

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

    if (XMRIG_OS_WIN)
        list(APPEND SOURCES src/hw/dmi/DmiReader_win.cpp)
    elseif(XMRIG_OS_LINUX)
        list(APPEND SOURCES src/hw/dmi/DmiReader_unix.cpp)
    endif()
else()
    remove_definitions(/DXMRIG_FEATURE_DMI)
endif()
