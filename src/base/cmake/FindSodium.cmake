find_path(
    sodium_INCLUDE_DIR
    NAMES sodium.h
    PATHS "${XMRIG_DEPS}" ENV "XMRIG_DEPS"
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
)

find_path(sodium_INCLUDE_DIR NAMES sodium.h)

find_library(
    sodium_LIBRARY
    NAMES libsodium.a sodium libsodium
    PATHS "${XMRIG_DEPS}" ENV "XMRIG_DEPS"
    PATH_SUFFIXES "lib"
    NO_DEFAULT_PATH
)

find_library(sodium_LIBRARY NAMES libsodium.a sodium libsodium)

set(SODIUM_LIBRARIES ${sodium_LIBRARY})
set(SODIUM_INCLUDE_DIRS ${sodium_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sodium DEFAULT_MSG sodium_LIBRARY sodium_INCLUDE_DIR)
