# This is a modified version of the `record.py` script from the LeRobot project (commit 459c95197ba1114ac8c3f538786ede005598f5e9).
# 'record' and 'record_loop' functions are modified.
# Added 'return_to_home_position' and 'is_close_to_home_position' functions.

import logging
import time
from dataclasses import asdict
from pprint import pformat
from typing import Dict

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    predict_action,
)
from lerobot.common.utils.utils import (
    init_logging,
    log_say,
    get_safe_torch_device,
)

from lerobot.record import RecordConfig
from lerobot.common.robots.robot import Robot
from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.utils.robot_utils import busy_wait


RECORDED_VIDEO_PATH = "videos/chunk-000/observation.images.front/episode_000000.mp4"
HOME_POSITION = {
    "shoulder_pan.pos": -1.8317757009345854,
    "shoulder_lift.pos": -95.53119730185497,
    "elbow_flex.pos": 99.45676776822091,
    "wrist_flex.pos": 42.48962655601659,
    "wrist_roll.pos": 1.6361416361416303,
    "gripper.pos": 1.5460295151089247,
}


def return_to_home_position(
    robot: Robot, home_position: Dict[str, float], step_size: float = 1.6
):
    is_home_position = False
    while not is_home_position:
        observation = robot.get_observation()
        action = {}
        joint_close_to_home = []
        for key, target_value in home_position.items():
            current_value = observation.get(key)
            diff = target_value - current_value
            if abs(diff) < 1:
                action[key] = target_value
                joint_close_to_home.append(True)
            else:
                diff = min(diff, step_size) if diff >= 0 else max(diff, -step_size)
                action[key] = current_value + diff
                joint_close_to_home.append(False)
        robot.send_action(action)
        if all(joint_close_to_home):
            is_home_position = True
        time.sleep(0.05)


def is_close_to_home_position(
    observation: Dict[str, float], home_position: Dict[str, float], threshold: float = 9
) -> bool:
    for key, target_value in home_position.items():
        current_value = observation.get(key)
        if abs(current_value - target_value) > threshold:
            return False
    return True


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    policy: PreTrainedPolicy,
    dataset: LeRobotDataset | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(
            f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps})."
        )

    # if policy is given it needs cleaning up
    # if policy is not None:
    policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        observation = robot.get_observation()
        # a = {key: item for key, item in observation.items() if key != "front"}
        # print(f"Current observation: {a}")

        if timestamp > control_time_s * 0.25 and is_close_to_home_position(
            observation=observation, home_position=HOME_POSITION
        ):
            return_to_home_position(robot=robot, home_position=HOME_POSITION)
            break

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(
                dataset.features, observation, prefix="observation"
            )
            # print(f"Current observation: {observation_frame['observation.state']}")

        # if policy is not None:
        action_values = predict_action(
            observation_frame,
            policy,
            get_safe_torch_device(policy.config.device),
            policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )
        action = {
            key: action_values[i].item() for i, key in enumerate(robot.action_features)
        }
        # else:
        #     action = teleop.get_action()

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        # print(f"Action to send: {action}")
        sent_action = robot.send_action(action)

        if dataset is not None:
            action_frame = build_dataset_frame(
                dataset.features, sent_action, prefix="action"
            )
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)

        # if display_data:
        #     for obs, val in observation.items():
        #         if isinstance(val, float):
        #             rr.log(f"observation.{obs}", rr.Scalar(val))
        #         elif isinstance(val, np.ndarray):
        #             rr.log(f"observation.{obs}", rr.Image(val), static=True)
        #     for act, val in action.items():
        #         if isinstance(val, float):
        #             rr.log(f"action.{act}", rr.Scalar(val))

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break


def record(cfg: RecordConfig, robot: Robot) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    # if cfg.display_data:
    #     _init_rerun(session_name="recording")

    # robot = make_robot_from_config(cfg.robot)
    # teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    action_features = hw_to_dataset_features(
        robot.action_features, "action", cfg.dataset.video
    )
    obs_features = hw_to_dataset_features(
        robot.observation_features, "observation", cfg.dataset.video
    )
    dataset_features = {**action_features, **obs_features}

    # Create empty dataset or load existing saved episodes
    sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
        * len(robot.cameras),
    )

    # Load pretrained policy
    # policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)

    # My comment: connect and disconnect robot outside this function
    # robot.connect()
    # if teleop is not None:
    #     teleop.connect()

    listener, events = init_keyboard_listener()

    for recorded_episodes in range(cfg.dataset.num_episodes):
        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        print(f"Recording episode {dataset.num_episodes}...")
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            # teleop=teleop,
            policy=policy,
            dataset=dataset,
            control_time_s=cfg.dataset.episode_time_s,
            single_task=cfg.dataset.single_task,
            display_data=cfg.display_data,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.dataset.num_episodes - 1)
            or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            print("Reset the environment...")
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                # teleop=teleop,
                control_time_s=cfg.dataset.reset_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()

        if events["stop_recording"]:
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    # robot.disconnect()
    # teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    # if cfg.dataset.push_to_hub:
    #     dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset
