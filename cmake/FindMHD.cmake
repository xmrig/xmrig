# - Try to find MHD
# Once done this will define
#
#  MHD_FOUND - system has MHD
#  MHD_INCLUDE_DIRS - the MHD include directory
#  MHD_LIBRARY - Link these to use MHD

find_path(
    MHD_INCLUDE_DIR
    NAMES microhttpd.h
    DOC "microhttpd include dir"
)

find_library(
    MHD_LIBRARY
    NAMES microhttpd microhttpd-10 libmicrohttpd libmicrohttpd-dll
    DOC "microhttpd library"
)

set(MHD_INCLUDE_DIRS ${MHD_INCLUDE_DIR})
set(MHD_LIBRARIES ${MHD_LIBRARY})

# debug library on windows
# same naming convention as in qt (appending debug library with d)
# boost is using the same "hack" as us with "optimized" and "debug"
# official MHD project actually uses _d suffix
if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    find_library(
        MHD_LIBRARY_DEBUG
        NAMES microhttpd_d microhttpd-10_d libmicrohttpd_d libmicrohttpd-dll_d
        DOC "mhd debug library"
    )
    set(MHD_LIBRARIES optimized ${MHD_LIBRARIES} debug ${MHD_LIBRARY_DEBUG})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mhd DEFAULT_MSG MHD_INCLUDE_DIR MHD_LIBRARY)
mark_as_advanced(MHD_INCLUDE_DIR MHD_LIBRARY)

