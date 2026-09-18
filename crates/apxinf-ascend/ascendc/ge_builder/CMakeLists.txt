cmake_minimum_required(VERSION 3.16)
project(ge_builder)

set(CMAKE_CXX_STANDARD 17)
# CANN's libgraph_base is built with the pre-cxx11 std::string ABI
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# pure host-side shared lib: generic graph construction + model build +
# load/save + execute. Links the GE/metadef libs shipped with the CANN
# toolkit (same link set as ge_poc).
set(GE_LIB_HINTS
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
find_library(GRAPH_LIB NAMES graph REQUIRED PATHS ${GE_LIB_HINTS})
find_library(GRAPH_BASE NAMES graph_base REQUIRED PATHS ${GE_LIB_HINTS})
find_library(GE_RUNNER NAMES ge_runner REQUIRED PATHS ${GE_LIB_HINTS})
find_library(GE_COMPILER NAMES ge_compiler REQUIRED PATHS ${GE_LIB_HINTS})
find_library(ACL_RT NAMES ascendcl REQUIRED PATHS ${GE_LIB_HINTS})

add_library(ge_builder SHARED ge_builder.cpp)
target_include_directories(ge_builder PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_libraries(ge_builder PRIVATE ${GRAPH_LIB} ${GRAPH_BASE} ${GE_RUNNER} ${GE_COMPILER} ${ACL_RT})
install(TARGETS ge_builder LIBRARY DESTINATION lib)

# standalone driver: same MatMul-pair chain as the ge_poc POC, but defined
# through the generic FFI (proves the layer) + a transposed-B variant
# (C1-②: MatMul transpose_x2/transpose_b + [n,k] weight descs, the layout
# the engine's NzCache pipeline produces)
add_executable(geb_main ge_builder_main.cpp)
target_include_directories(geb_main PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_libraries(geb_main PRIVATE ge_builder ${GRAPH_LIB} ${GRAPH_BASE} ${GE_RUNNER} ${GE_COMPILER} ${ACL_RT})
install(TARGETS geb_main RUNTIME DESTINATION bin)
