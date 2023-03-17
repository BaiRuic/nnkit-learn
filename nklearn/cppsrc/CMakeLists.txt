cmake_minimum_required(VERSION 3.2)
project(needle C CXX)

# find correct version of Python
execute_process(COMMAND python3-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
# if not find pybind in root, just set pybind11_DIR manually.
if(__pybind_exit_code AND NOT __pybind_exit_code EQUAL 0)
  set(pybind11_DIR /home/brc/miniconda3/envs/dlsys/lib/python3.8/site-packages/pybind11/share/cmake/pybind11)
endif()

find_package(pybind11 PATHS ${pybind11_DIR})


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE ./ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../ndarray/
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()