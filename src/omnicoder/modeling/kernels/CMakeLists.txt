cmake_minimum_required(VERSION 3.18)
project(omnicoder_dml_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED TORCH_DIR)
  message(STATUS "Using PyTorch CMake integration from Python site-packages")
endif()

find_package(Torch REQUIRED)

add_library(omnicoder_dml_native MODULE dml_fused_attention.cpp)
target_link_libraries(omnicoder_dml_native PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(omnicoder_dml_native PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
if (MSVC)
  target_compile_options(omnicoder_dml_native PRIVATE /O2)
else()
  target_compile_options(omnicoder_dml_native PRIVATE -O3)
endif()

# On Windows, produce a .pyd extension
set_target_properties(omnicoder_dml_native PROPERTIES PREFIX "" OUTPUT_NAME "omnicoder_dml_native")

if (WIN32)
  # Link Direct3D12 and DXGI; DirectML is header-only and loads at runtime
  target_link_libraries(omnicoder_dml_native PRIVATE d3d12 dxgi)
  target_compile_definitions(omnicoder_dml_native PRIVATE OMNICODER_BUILD_DML=1)
endif()


