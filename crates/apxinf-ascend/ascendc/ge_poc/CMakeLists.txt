cmake_minimum_required(VERSION 3.16)
project(ge_matmul_poc)

set(CMAKE_CXX_STANDARD 17)
# CANN's libgraph_base is built with the pre-cxx11 std::string ABI
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# pure host-side shared lib: graph construction + model build + execute.
# Links the GE/metadef libs shipped with the CANN toolkit.
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

add_library(ge_matmul_poc SHARED ge_matmul_poc.cpp)
target_include_directories(ge_matmul_poc PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_libraries(ge_matmul_poc PRIVATE ${GRAPH_LIB} ${GRAPH_BASE} ${GE_RUNNER} ${GE_COMPILER} ${ACL_RT})
install(TARGETS ge_matmul_poc LIBRARY DESTINATION lib)

add_executable(ge_poc_main ge_poc_main.cpp)
target_include_directories(ge_poc_main PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_libraries(ge_poc_main PRIVATE ge_matmul_poc ${GRAPH_LIB} ${GRAPH_BASE} ${GE_RUNNER} ${GE_COMPILER} ${ACL_RT})
install(TARGETS ge_poc_main RUNTIME DESTINATION bin)
