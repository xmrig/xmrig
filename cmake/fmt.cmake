# SPDX-FileCopyrightText: © 2023 Jean-Pierre De Jesus DIAZ <me@jeandudey.tech>
# SPDX-License-Identifier: GPL-3.0-or-later


if(WITH_BUNDLED_FMT)
    add_library(fmt INTERFACE)
    target_sources(fmt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/fmt/format.cc)
    target_include_directories(fmt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/fmt)
else()
    set(FMT_LIBRARY fmt)
    find_package(fmt REQUIRED)
    set(FMT_LIBRARY fmt::fmt)
endif()
