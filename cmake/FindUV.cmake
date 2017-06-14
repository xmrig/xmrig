find_path(UV_INCLUDE_DIR NAMES uv.h)
find_library(UV_LIBRARY NAMES uv libuv)

set(UV_LIBRARIES ${UV_LIBRARY})
set(UV_INCLUDE_DIRS ${UV_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UV DEFAULT_MSG UV_LIBRARY UV_INCLUDE_DIR)
