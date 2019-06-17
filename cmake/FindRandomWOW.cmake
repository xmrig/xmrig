find_path(
    RANDOMWOW_INCLUDE_DIR
    NAMES randomwow.h
    PATHS "${XMRIG_DEPS}" ENV "XMRIG_DEPS"
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
)

find_path(RANDOMWOW_INCLUDE_DIR NAMES randomwow.h)

find_library(
    RANDOMWOW_LIBRARY
    NAMES librandomwow.a randomwow librandomwow
    PATHS "${XMRIG_DEPS}" ENV "XMRIG_DEPS"
    PATH_SUFFIXES "lib"
    NO_DEFAULT_PATH
)

find_library(RANDOMWOW_LIBRARY NAMES librandomwow.a randomwow librandomwow)

set(RANDOMWOW_LIBRARIES ${RANDOMWOW_LIBRARY})
set(RANDOMWOW_INCLUDE_DIRS ${RANDOMWOW_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RANDOMWOW DEFAULT_MSG RANDOMWOW_LIBRARY RANDOMWOW_INCLUDE_DIR)
