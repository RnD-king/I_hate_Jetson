cmake_minimum_required(VERSION 3.18)
project(vision)

# -------------------------------------------------------------
# CUDA & C++
# -------------------------------------------------------------
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Jetson Orin Nano (Compute Capability 8.7)
set(CMAKE_CUDA_ARCHITECTURES 87)

# -------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

include_directories(
  include
  ${CUDAToolkit_INCLUDE_DIRS}
)

# -------------------------------------------------------------
# 1) CUDA Preprocess Library
# -------------------------------------------------------------
add_library(cuda_preprocess STATIC
  src/yolo_preprocess.cu
)
set_target_properties(cuda_preprocess PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_preprocess
  CUDA::cudart
)

# -------------------------------------------------------------
# 2) Line Perception Node (pipeline modules)
# -------------------------------------------------------------
add_executable(line_perception_node
  src/line_perception_node.cpp
  src/yolo_trt_engine.cpp
  src/line_point_extractor.cpp
  src/coordinate_rectifier.cpp
  src/feature_extractor.cpp
)

target_include_directories(line_perception_node PUBLIC
  include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# (중요) OpenCV 헤더는 시스템 4.5 경로로 고정
target_include_directories(line_perception_node SYSTEM PUBLIC
  /usr/include/opencv4
)

# (중요) OpenCV 4.5를 절대경로로 강제 링크 (혼용 차단 핵심)
# 필요에 따라 highgui/videoio가 필요하면 같은 방식으로 추가하십시오.
target_link_libraries(line_perception_node
  cuda_preprocess

  /lib/aarch64-linux-gnu/libopencv_core.so.4.5d
  /lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5d
  /lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5d

  nvinfer
  nvinfer_plugin
  cudart
)

ament_target_dependencies(line_perception_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)

install(TARGETS line_perception_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
