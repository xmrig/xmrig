include (src/backend/cpu/cpu.cmake)
include (src/backend/opencl/opencl.cmake)
include (src/backend/common/common.cmake)


set(HEADERS_BACKEND
    "${HEADERS_BACKEND_COMMON}"
    "${HEADERS_BACKEND_CPU}"
    "${HEADERS_BACKEND_OPENCL}"
   )

set(SOURCES_BACKEND
    "${SOURCES_BACKEND_COMMON}"
    "${SOURCES_BACKEND_CPU}"
    "${SOURCES_BACKEND_OPENCL}"
   )
