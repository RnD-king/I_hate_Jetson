from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.g1_vision.g1_vision_config import G1VisionRoughCfg, G1VisionRoughCfgPPO
from legged_gym.envs.g1_vision.g1_vision_env import G1VisionRobot
from .base.legged_robot import LeggedRobot

try:
    from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

    _has_go2 = True
except ModuleNotFoundError:
    _has_go2 = False

try:
    from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
    from legged_gym.envs.h1.h1_env import H1Robot

    _has_h1 = True
except ModuleNotFoundError:
    _has_h1 = False

try:
    from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
    from legged_gym.envs.h1_2.h1_2_env import H1_2Robot

    _has_h1_2 = True
except ModuleNotFoundError:
    _has_h1_2 = False
try:
    from legged_gym.envs.g1_highlevel.g1_highlevel_config import G1HighLevelCfg, G1HighLevelCfgPPO
    from legged_gym.envs.g1_highlevel.g1_highlevel_env import G1HighLevelEnv

    _has_g1_highlevel = True
except ModuleNotFoundError:
    _has_g1_highlevel = False

try:
    from legged_gym.envs.g1_highlevel_v2.g1_highlevel_v2_config import (
        G1HighLevelV2Cfg,
        G1HighLevelV2CfgPPO,
    )
    from legged_gym.envs.g1_highlevel_v2.g1_highlevel_v2_env import G1HighLevelV2Env

    _has_g1_highlevel_v2 = True
except ModuleNotFoundError:
    _has_g1_highlevel_v2 = False

from legged_gym.utils.task_registry import task_registry

if _has_go2:
    task_registry.register("go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
if _has_h1:
    task_registry.register("h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
if _has_h1_2:
    task_registry.register("h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register("g1_vision", G1VisionRobot, G1VisionRoughCfg(), G1VisionRoughCfgPPO())
if _has_g1_highlevel:
    task_registry.register("g1_highlevel", G1HighLevelEnv, G1HighLevelCfg(), G1HighLevelCfgPPO())
if _has_g1_highlevel_v2:
    task_registry.register("g1_highlevel_v2", G1HighLevelV2Env, G1HighLevelV2Cfg(), G1HighLevelV2CfgPPO())
