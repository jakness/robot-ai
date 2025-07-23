from lerobot.record import RecordConfig, DatasetRecordConfig
from lerobot.common.robots.utils import make_robot_from_config
from lerobot.common.robots.so100_follower.config_so100_follower import (
    SO100FollowerConfig,
)
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.policies.act.configuration_act import ACTConfig


BASE_CONFIG = RecordConfig(
    robot=SO100FollowerConfig(
        port="COM3",
        id="so100_follower",
        cameras={
            "front": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30)
        },
    ),
    dataset=DatasetRecordConfig(
        repo_id="test/eval_robot",
        push_to_hub=False,
        single_task="Evaluate",
        fps=30,
        num_episodes=1,
        reset_time_s=3,
    ),
    policy=ACTConfig(
        device="cuda",
        chunk_size=60,
        n_action_steps=1,
        temporal_ensemble_coeff=0.01,
    ),
)
ROBOT = make_robot_from_config(BASE_CONFIG.robot)
