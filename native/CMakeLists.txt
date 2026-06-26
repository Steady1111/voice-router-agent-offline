cmake_minimum_required(VERSION 3.16)
project(voice_router_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to a sherpa-onnx build tree or install prefix, e.g.:
#   cmake -B build -DSHERPA_ONNX_ROOT=/path/to/sherpa-onnx/build/install
set(SHERPA_ONNX_ROOT "" CACHE PATH "sherpa-onnx install prefix containing include/ and lib/")

if(NOT SHERPA_ONNX_ROOT)
  message(FATAL_ERROR "Set -DSHERPA_ONNX_ROOT= to sherpa-onnx install prefix")
endif()

include_directories(${SHERPA_ONNX_ROOT}/include)

find_library(SHERPA_ONNX_LIB sherpa-onnx-c-api
  PATHS ${SHERPA_ONNX_ROOT}/lib ${SHERPA_ONNX_ROOT}/lib64
  NO_DEFAULT_PATH)

if(NOT SHERPA_ONNX_LIB)
  message(FATAL_ERROR "libsherpa-onnx-c-api not found under ${SHERPA_ONNX_ROOT}/lib")
endif()

add_executable(voice-router-asr-worker asr_worker.cpp)
target_link_libraries(voice-router-asr-worker PRIVATE ${SHERPA_ONNX_LIB})

get_filename_component(SHERPA_ONNX_LIB_DIR "${SHERPA_ONNX_LIB}" DIRECTORY)
set_target_properties(voice-router-asr-worker PROPERTIES
  BUILD_RPATH "${SHERPA_ONNX_LIB_DIR}"
  INSTALL_RPATH "${SHERPA_ONNX_LIB_DIR}"
)
