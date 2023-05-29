# SPDX-FileCopyrightText: © 2023 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
# SPDX-License-Identifier: GPL-3.0-or-later

find_package(fmt QUIET)

if(fmt_FOUND)
    set(FMT_LIBRARY fmt::fmt)
else()
    message(STATUS "Using bundled fmt library")
    add_library(fmt INTERFACE)
    target_sources(fmt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/fmt/format.cc)
    target_include_directories(fmt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/fmt)
    set(FMT_LIBRARY fmt)
endif()
