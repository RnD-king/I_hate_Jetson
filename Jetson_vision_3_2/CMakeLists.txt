cmake_minimum_required(VERSION 3.18)
project(vision)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# CUDA
# =========================
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# (1) CUDA 아키텍처 자동 설정: 빌드 머신 GPU에 맞춤
# - Jetson Orin Nano => 보통 87로 자동 설정됨
# - Desktop RTX30    => 86 등 자동 설정됨
# 주의: CMake 3.18+에서 지원되는 방식으로 사용
set(CMAKE_CUDA_ARCHITECTURES 87)

# (3) CUDA 경로 FORCE 제거: 시스템/JetPack 환경에 맡김
find_package(CUDAToolkit REQUIRED)

# =========================
# ROS 2 / deps
# =========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# OpenCV는 cv_bridge 전이를 기대 (명시 find_package(OpenCV ...) 없음)
message(STATUS "OpenCV_DIR=${OpenCV_DIR}")
message(STATUS "OpenCV_VERSION=${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")

# =========================
# TensorRT
# =========================
set(TENSORRT_ROOT "" CACHE PATH "TensorRT root directory (optional)")

# (2) aarch64-linux-gnu 경로 추가 + 기존 x86_64-linux-gnu 유지
find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${TENSORRT_ROOT} /usr /usr/local
  PATH_SUFFIXES
    include
    include/aarch64-linux-gnu
    include/x86_64-linux-gnu
)

find_library(TENSORRT_NVINFER
  NAMES nvinfer
  HINTS ${TENSORRT_ROOT} /usr /usr/local
  PATH_SUFFIXES
    lib
    lib64
    lib/aarch64-linux-gnu
    lib/x86_64-linux-gnu
)

find_library(TENSORRT_NVINFER_PLUGIN
  NAMES nvinfer_plugin
  HINTS ${TENSORRT_ROOT} /usr /usr/local
  PATH_SUFFIXES
    lib
    lib64
    lib/aarch64-linux-gnu
    lib/x86_64-linux-gnu
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

# =========================
# Core library
# =========================
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

# =========================
# Node executable
# =========================
add_executable(line_perception_node
  src/line_perception_node.cpp
)

target_link_libraries(line_perception_node
  vision_core
)

ament_target_dependencies(line_perception_node
  rclcpp sensor_msgs geometry_msgs std_msgs cv_bridge
)

# =========================
# Install
# =========================
install(TARGETS
  vision_core
  line_perception_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
