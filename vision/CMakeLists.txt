cmake_minimum_required(VERSION 3.18)
project(vision)

# ============================================================
# IMPORTANT:
#  - Your OpenCV (/usr/local) calls find_package(CUDA 12.4) inside OpenCVConfig.cmake.
#  - Many local CUDA installs DO NOT provide CUDAConfig.cmake.
#  - Therefore we MUST allow legacy FindCUDA.cmake behavior here.
# ============================================================
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)  # allow FindCUDA
endif()

# ---------- 기본 설정 ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 사용 (.cu 컴파일 필요)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# (RTX 3060 = 8.6) / (Jetson Orin = 8.7)
set(CMAKE_CUDA_ARCHITECTURES 86)

# ============================================================
# FindCUDA hints (MUST be set BEFORE cv_bridge/OpenCV)
# These variables help OpenCVConfig.cmake -> find_package(CUDA 12.4)
# ============================================================
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.4" CACHE PATH "CUDA toolkit root (for FindCUDA/OpenCV)" FORCE)
set(CUDA_BIN_PATH         "/usr/local/cuda-12.4" CACHE PATH "CUDA bin path (for FindCUDA/OpenCV)" FORCE)

# ---------- ROS2 ----------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# cv_bridge가 내부에서 OpenCV(OpenCVConfig.cmake)를 부르고,
# 그 OpenCVConfig.cmake가 find_package(CUDA 12.4)를 호출할 수 있음
find_package(cv_bridge REQUIRED)

# ---------- OpenCV ----------
find_package(OpenCV REQUIRED)

# ---------- CUDA Toolkit (our own targets) ----------
# This is for *our linking* (CUDA::cudart). Independent from OpenCV's FindCUDA call.
find_package(CUDAToolkit REQUIRED)

# ---------- TensorRT ----------
# (optional) set this if TensorRT is in a non-standard place:
set(TENSORRT_ROOT "" CACHE PATH "TensorRT root directory (optional)")

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS
    ${TENSORRT_ROOT}
    /usr
    /usr/local
    /usr/include/x86_64-linux-gnu
  PATH_SUFFIXES include include/x86_64-linux-gnu
)

find_library(TENSORRT_NVINFER
  NAMES nvinfer
  HINTS
    ${TENSORRT_ROOT}
    /usr
    /usr/local
    /usr/lib/x86_64-linux-gnu
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

find_library(TENSORRT_NVINFER_PLUGIN
  NAMES nvinfer_plugin
  HINTS
    ${TENSORRT_ROOT}
    /usr
    /usr/local
    /usr/lib/x86_64-linux-gnu
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find TensorRT headers (NvInfer.h). Set TENSORRT_ROOT or install TensorRT dev package.")
endif()

if(NOT TENSORRT_NVINFER)
  message(FATAL_ERROR "Could not find TensorRT library: nvinfer. Set TENSORRT_ROOT or install TensorRT.")
endif()

if(NOT TENSORRT_NVINFER_PLUGIN)
  message(FATAL_ERROR "Could not find TensorRT library: nvinfer_plugin. Set TENSORRT_ROOT or install TensorRT.")
endif()

message(STATUS "CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA_BIN_PATH=${CUDA_BIN_PATH}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib nvinfer: ${TENSORRT_NVINFER}")
message(STATUS "TensorRT lib nvinfer_plugin: ${TENSORRT_NVINFER_PLUGIN}")

# ---------- 모듈 라이브러리 (cpp + cu 묶기) ----------
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
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
)

# ---------- 실행 노드 ----------
add_executable(line_perception_node
  src/line_perception_node.cpp
)

target_include_directories(line_perception_node PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(line_perception_node
  vision_core
)

ament_target_dependencies(line_perception_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
)

# ---------- 설치 ----------
install(TARGETS
  vision_core
  line_perception_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
