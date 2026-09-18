cmake_minimum_required(VERSION 3.16)
project(ada_rms_norm)

set(RUN_MODE "npu" CACHE STRING "run mode: cpu/sim/npu")
set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/ada_rms_norm_dynamic.cpp
)

if(RUN_MODE STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
else()
    # npu_lib.cmake invokes ascendc_library(KERNEL_FILES) itself
    include(cmake/npu_lib.cmake)

    add_library(ada_rms_norm_host SHARED ada_rms_norm_host.cpp)
    target_include_directories(ada_rms_norm_host PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/auto_gen
    )
    target_link_libraries(ada_rms_norm_host PRIVATE ascendc_runtime ascendc_kernels_${RUN_MODE})
    install(TARGETS ada_rms_norm_host LIBRARY DESTINATION lib)
endif()
