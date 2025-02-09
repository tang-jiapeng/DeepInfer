cmake_minimum_required(VERSION 3.16)

project(DeepInfer)
set(CMAKE_CXX_STANDARD 17)
find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)

if(!WIN32)
    set(link_lib "${link_lib} pthread")
endif()

set(link_lib glog::glog GTest::GTest)
set(link_math_lib  ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

aux_source_directory(./test DIR_TEST)
aux_source_directory(./src/data DIR_DATA)
aux_source_directory(./src/runtime DIR_RUNTIME)
aux_source_directory(./src/runtime/pnnx DIR_RUNTIME)

add_executable(DeepInfer main.cc ${DIR_TEST} ${DIR_DATA} ${DIR_RUNTIME})

target_link_libraries(DeepInfer ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(DeepInfer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(DeepInfer PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(DeepInfer PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(DeepInfer PUBLIC ./include)

enable_testing()