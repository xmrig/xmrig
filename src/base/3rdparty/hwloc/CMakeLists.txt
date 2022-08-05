cmake_minimum_required (VERSION  2.8.12)
project (hwloc C)

include_directories(include)
include_directories(src)

add_definitions(/D_CRT_SECURE_NO_WARNINGS)
set(CMAKE_C_FLAGS_RELEASE "/MT /O2 /Ob2 /DNDEBUG")

set(HEADERS
    include/hwloc.h
    src/static-components.h
    )

set(SOURCES
    src/base64.c
    src/bind.c
    src/bitmap.c
    src/components.c
    src/diff.c
    src/distances.c
    src/misc.c
    src/pci-common.c
    src/shmem.c
    src/topology.c
    src/topology-noos.c
    src/topology-synthetic.c
    src/topology-windows.c
    src/topology-x86.c
    src/topology-xml.c
    src/topology-xml-nolibxml.c
    src/traversal.c
    src/memattrs.c
    src/cpukinds.c
   )

add_library(hwloc STATIC
    ${HEADERS}
    ${SOURCES}
    )
