cmake_minimum_required (VERSION 2.8)
project (jansson C)

add_definitions(-DHAVE_CONFIG_H)

# Add the lib sources.
file(GLOB JANSSON_SRC *.c)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Os")

set(JANSSON_HDR_PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/hashtable.h
   ${CMAKE_CURRENT_SOURCE_DIR}/jansson_private.h
   ${CMAKE_CURRENT_SOURCE_DIR}/strbuffer.h
   ${CMAKE_CURRENT_SOURCE_DIR}/utf.h
   ${CMAKE_CURRENT_SOURCE_DIR}/jansson_private_config.h)

set(JANSSON_HDR_PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}/jansson_config.h
   ${CMAKE_CURRENT_SOURCE_DIR}/jansson.h)

add_library(jansson STATIC
    ${JANSSON_SRC}
    ${JANSSON_HDR_PRIVATE}
    ${JANSSON_HDR_PUBLIC})
