# Copyright 2023 Intel Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch realsense2_camera node."""
import os
import yaml
from launch import LaunchDescription
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


configurable_parameters = [{'name': 'camera_name',                  'default': 'camera', 'description': 'camera unique name'},
                           {'name': 'camera_namespace',             'default': '', 'description': 'namespace for camera'},
                           {'name': 'serial_no',                    'default': "''", 'description': 'choose device by serial number'},
                           {'name': 'usb_port_id',                  'default': "''", 'description': 'choose device by usb port id'},
                           {'name': 'device_type',                  'default': "''", 'description': 'choose device by type'},
                           {'name': 'config_file',                  'default': "''", 'description': 'yaml config file'},
                           {'name': 'json_file_path',               'default': "''", 'description': 'allows advanced configuration'},
                           {'name': 'initial_reset',                'default': 'false', 'description': "''"},
                           {'name': 'accelerate_gpu_with_glsl',     'default': "false", 'description': 'enable GPU acceleration with GLSL'},
                           {'name': 'rosbag_filename',              'default': "''", 'description': 'A realsense bagfile to run from as a device'},
                           {'name': 'log_level',                    'default': 'info', 'description': 'debug log level [DEBUG|INFO|WARN|ERROR|FATAL]'},
                           {'name': 'output',                       'default': 'screen', 'description': 'pipe node output [screen|log]'},
                           {'name': 'enable_color',                 'default': 'true', 'description': 'enable color stream'},
                           {'name': 'rgb_camera.color_profile',     'default': '640,480,15', 'description': 'color stream profile'},
                           {'name': 'rgb_camera.color_format',      'default': 'RGB8', 'description': 'color stream format'},
                           {'name': 'color_qos',                    'default': 'SENSOR_DATA', 'description': 'color image QoS'},
                           {'name': 'rgb_camera.enable_auto_exposure', 'default': 'true', 'description': 'enable/disable auto exposure for color image'}, # 자동 노출 끄기
                           {'name': 'rgb_camera.enable_auto_white_balance', 'default': 'true', 'description': 'enable/disable auto white balance for color image'}, # 자동 화밸 끄기
                           {'name': 'enable_depth',                 'default': 'true', 'description': 'enable depth stream'},
                           {'name': 'enable_infra',                 'default': 'false', 'description': 'enable infra0 stream'},
                           {'name': 'enable_infra1',                'default': 'false', 'description': 'enable infra1 stream'},
                           {'name': 'enable_infra2',                'default': 'false', 'description': 'enable infra2 stream'},
                           {'name': 'depth_module.depth_profile',   'default': '640,480,15', 'description': 'depth stream profile'},
                           {'name': 'depth_module.depth_format',    'default': 'Z16', 'description': 'depth stream format'},
                           {'name': 'depth_qos',                    'default': 'SENSOR_DATA', 'description': 'depth image QoS'},
                           {'name': 'depth_module.infra_profile',   'default': '640,480,15', 'description': 'infra streams (0/1/2) profile'},
                           {'name': 'depth_module.infra_format',    'default': 'RGB8', 'description': 'infra0 stream format'},
                           {'name': 'depth_module.infra1_format',   'default': 'Y8', 'description': 'infra1 stream format'},
                           {'name': 'depth_module.infra2_format',   'default': 'Y8', 'description': 'infra2 stream format'},
                           {'name': 'depth_module.exposure',        'default': '8500', 'description': 'Depth module manual exposure value'},
                           {'name': 'depth_module.gain',            'default': '16', 'description': 'Depth module manual gain value'},
                           {'name': 'depth_module.hdr_enabled',     'default': 'false', 'description': 'Depth module hdr enablement flag. Used for hdr_merge filter'},
                           {'name': 'depth_module.enable_auto_exposure', 'default': 'false', 'description': 'enable/disable auto exposure for depth image'},
                           {'name': 'depth_module.exposure.1',      'default': '7500', 'description': 'Depth module first exposure value. Used for hdr_merge filter'},
                           {'name': 'depth_module.gain.1',          'default': '16', 'description': 'Depth module first gain value. Used for hdr_merge filter'},
                           {'name': 'depth_module.exposure.2',      'default': '1', 'description': 'Depth module second exposure value. Used for hdr_merge filter'},
                           {'name': 'depth_module.gain.2',          'default': '16', 'description': 'Depth module second gain value. Used for hdr_merge filter'},
                           {'name': 'enable_sync',                  'default': 'false', 'description': "'enable sync mode'"},  # 동기화 켜기
                           {'name': 'enable_rgbd',                  'default': 'false', 'description': "'enable rgbd topic'"},
                           {'name': 'enable_gyro',                  'default': 'false', 'description': "'enable gyro stream'"},
                           {'name': 'enable_accel',                 'default': 'false', 'description': "'enable accel stream'"},
                           {'name': 'gyro_fps',                     'default': '0', 'description': "''"},
                           {'name': 'gyro_qos',                     'default': 'SENSOR_DATA', 'description': 'gyro QoS'},
                           {'name': 'accel_fps',                    'default': '0', 'description': "''"},
                           {'name': 'accel_qos',                    'default': 'SENSOR_DATA', 'description': 'accel QoS'},
                           {'name': 'unite_imu_method',             'default': "0", 'description': '[0-None, 1-copy, 2-linear_interpolation]'},
                           {'name': 'clip_distance',                'default': '-2.', 'description': "''"},
                           {'name': 'angular_velocity_cov',         'default': '0.01', 'description': "''"},
                           {'name': 'linear_accel_cov',             'default': '0.01', 'description': "''"},
                           {'name': 'diagnostics_period',           'default': '0.0', 'description': 'Rate of publishing diagnostics. 0=Disabled'},
                           {'name': 'publish_tf',                   'default': 'true', 'description': '[bool] enable/disable publishing static & dynamic TF'},
                           {'name': 'tf_publish_rate',              'default': '0.0', 'description': '[double] rate in Hz for publishing dynamic TF'},
                           {'name': 'pointcloud.enable',            'default': 'false', 'description': ''}, # 포인트클라우드
                           {'name': 'pointcloud.stream_filter',     'default': '2', 'description': 'texture stream for pointcloud'},
                           {'name': 'pointcloud.stream_index_filter','default': '0', 'description': 'texture stream index for pointcloud'},
                           {'name': 'pointcloud.ordered_pc',        'default': 'true', 'description': ''},
                           {'name': 'pointcloud.allow_no_texture_points', 'default': 'true', 'description': "''"},
                           {'name': 'align_depth.enable',           'default': 'false', 'description': 'enable align depth filter'},  # 자동 정렬 켜기
                           {'name': 'colorizer.enable',             'default': 'false', 'description': 'enable colorizer filter'},
                           {'name': 'decimation_filter.enable',     'default': 'false', 'description': 'enable_decimation_filter'},
                           {'name': 'spatial_filter.enable',        'default': 'false', 'description': 'enable_spatial_filter'},
                           {'name': 'temporal_filter.enable',       'default': 'false', 'description': 'enable_temporal_filter'},
                           {'name': 'disparity_filter.enable',      'default': 'false', 'description': 'enable_disparity_filter'},
                           {'name': 'hole_filling_filter.enable',   'default': 'false', 'description': 'enable_hole_filling_filter'},
                           {'name': 'hdr_merge.enable',             'default': 'false', 'description': 'hdr_merge filter enablement flag'},
                           {'name': 'wait_for_device_timeout',      'default': '-1.', 'description': 'Timeout for waiting for device to connect (Seconds)'},
                           {'name': 'reconnect_timeout',            'default': '6.', 'description': 'Timeout(seconds) between consequtive reconnection attempts'},
                          ]

def declare_configurable_parameters(parameters):
    return [
        DeclareLaunchArgument(
            param['name'],
            default_value=param['default'],
            description=param['description']
        )
        for param in parameters
    ]

def set_configurable_parameters(parameters):
    return {
        param['name']: LaunchConfiguration(param['name'])
        for param in parameters
    }

def yaml_to_dict(path_to_yaml):
    with open(path_to_yaml, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def launch_setup(context, params, param_name_suffix=''):
    _config_file = LaunchConfiguration('config_file' + param_name_suffix).perform(context)
    params_from_file = {} if _config_file == "''" else yaml_to_dict(_config_file)

    _output = LaunchConfiguration('output' + param_name_suffix)
    if os.getenv('ROS_DISTRO') == 'foxy':
        _output = context.perform_substitution(_output)

    return [
        launch_ros.actions.Node(
            package='realsense2_camera',
            namespace=LaunchConfiguration('camera_namespace' + param_name_suffix),
            name=LaunchConfiguration('camera_name' + param_name_suffix),
            executable='realsense2_camera_node',
            parameters=[params, params_from_file, {
                'rgb_camera.enable_auto_exposure': False,
                'rgb_camera.exposure': 155,
                'rgb_camera.enable_auto_white_balance': False,
                'rgb_camera.white_balance': 4200,

                'rgb_camera.saturation': 55,
                'rgb_camera.contrast': 55,
                'rgb_camera.sharpness': 30,
                'rgb_camera.gamma': 300, 
                'rgb_camera.gain': 64,

                # 'enable_gyro': True,
                # 'enable_accel': True,
                # 'gyro_fps': 200,
                # 'accel_fps': 200,

                # 'publish_tf': True,
                # 'publish_imu_tf': True,
            }], 
            output=_output,
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level' + param_name_suffix)],
            emulate_tty=True,
        )
    ]

def generate_launch_description():
    return LaunchDescription(
        declare_configurable_parameters(configurable_parameters) + [
            OpaqueFunction(function=launch_setup, kwargs={'params': set_configurable_parameters(configurable_parameters)})
        ]
    )


# gain
# 높이면 전체 이미지가 밝아집니다. 대신 센서 신호를 증폭하는 거라 노이즈도 같이 늘고, 어두운 영역이 지저분해질 수 있습니다. 너무 높이면 빨간 바닥의 미세한 얼룩이나 그림자까지 튀어서 오검출 원인이 될 수 있습니다. 낮추면 노이즈는 줄지만 전체가 어두워지고, 주황 공/노란 허들이 배경에 묻힐 수 있습니다.
# gamma
# 중간 밝기 영역을 휘게 만드는 값입니다. 노출처럼 모든 밝기를 똑같이 올리는 게 아니라, 어두운 부분과 중간톤의 보이는 정도를 바꿉니다. 높이면 보통 중간톤/어두운 부분이 더 떠 보일 수 있고, 낮추면 중간톤이 눌려서 대비가 강해 보일 수 있습니다. 너무 건드리면 HSV의 V 분포가 바뀌어서 학습/추론 색 분포가 흔들립니다.
# contrast
# 높이면 밝은 곳은 더 밝고 어두운 곳은 더 어둡게 보입니다. 경계가 선명해져서 흰 점선/허들 테두리는 잘 보일 수 있지만, 빛번짐은 더 하얗게 날아가고 어두운 빨간 바닥은 검게 뭉칠 수 있습니다. 낮추면 날아감과 검은 뭉침은 줄지만 화면이 밋밋해지고, 흰 점선과 빛번짐의 차이가 줄어들 수 있습니다.
# saturation
# 높이면 색이 진해집니다. 빨간 바닥, 노란 허들, 주황 공이 강하게 보이지만, 너무 높이면 빨강/주황/노랑이 서로 가까워져서 색 분리가 오히려 나빠질 수 있습니다. 낮추면 색이 옅어지고 조명 변화에는 조금 둔감해질 수 있지만, 너무 낮으면 노란 허들/주황 공이 배경과 구분이 약해집니다.
# sharpness
# 높이면 윤곽과 작은 패턴이 강조됩니다. 흰 점선 경계, 허들 모서리, 공 테두리가 더 또렷해질 수 있습니다. 대신 노이즈, 바닥 무늬, 빛번짐 가장자리도 같이 강조돼서 오검출이 늘 수 있습니다. 낮추면 화면이 부드러워지고 노이즈가 덜 튀지만 작은 점선이나 얇은 경계가 흐려질 수 있습니다.

# exposure -> white_balance -> saturation -> contrast -> sharpness -> gamma/gain
