cmake_minimum_required(VERSION 3.18)
project(vision)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architecture, e.g. 86 for RTX Ampere, 87 for Jetson Orin")
endif()

if(EXISTS "/usr/local/cuda-12.4")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.4" CACHE PATH "CUDA toolkit root")
  set(CUDA_BIN_PATH         "/usr/local/cuda-12.4" CACHE PATH "CUDA bin path")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV_DIR=${OpenCV_DIR}")
message(STATUS "OpenCV_VERSION=${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")

find_package(CUDAToolkit REQUIRED)

# ---------- TensorRT ----------
set(TENSORRT_ROOT "" CACHE PATH "TensorRT root directory (optional)")

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${TENSORRT_ROOT} /usr /usr/local /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu
  PATH_SUFFIXES include include/x86_64-linux-gnu
)

find_library(TENSORRT_NVINFER
  NAMES nvinfer
  HINTS ${TENSORRT_ROOT} /usr /usr/local /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

find_library(TENSORRT_NVINFER_PLUGIN
  NAMES nvinfer_plugin
  HINTS ${TENSORRT_ROOT} /usr /usr/local /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find TensorRT headers (NvInfer.h).")
endif()
if(NOT TENSORRT_NVINFER)
  message(FATAL_ERROR "Could not find TensorRT library: nvinfer.")
endif()
if(NOT TENSORRT_NVINFER_PLUGIN)
  message(FATAL_ERROR "Could not find TensorRT library: nvinfer_plugin.")
endif()

add_library(vision_core
  src/yolo_trt_engine.cpp
  src/line_point_extractor.cpp
  src/coordinate_rectifier.cpp
  src/feature_extractor.cpp
  src/yolo_preprocess.cu
)

set_target_properties(vision_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(vision_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(vision_core
  ${OpenCV_LIBS}
  ${TENSORRT_NVINFER}
  ${TENSORRT_NVINFER_PLUGIN}
  CUDA::cudart
)

ament_target_dependencies(vision_core
  rclcpp sensor_msgs geometry_msgs std_msgs cv_bridge
)

add_executable(line_perception_node
  src/line_perception_node.cpp
)

target_link_libraries(line_perception_node
  vision_core
)

ament_target_dependencies(line_perception_node
  rclcpp sensor_msgs geometry_msgs std_msgs cv_bridge
)

# ============================================================
# NEW: centers_to_features_node
# ============================================================
add_executable(centers_to_features_node
  src/centers_to_features_node.cpp
)

target_link_libraries(centers_to_features_node
  vision_core
)

ament_target_dependencies(centers_to_features_node
  rclcpp geometry_msgs std_msgs
)

install(TARGETS
  vision_core
  line_perception_node
  centers_to_features_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
