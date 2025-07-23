import os
import shutil
from typing import List
from copy import deepcopy
from pathlib import Path

from agents import function_tool
from google import genai

from robot_ai.constants import GEMINI_API_KEY, TOOL_EXECUTION_VIDEO_DIR_PATH
from robot_ai.lerobot_utils.record import (
    record,
    return_to_home_position,
    HOME_POSITION,
    RECORDED_VIDEO_PATH,
)
from robot_ai.lerobot_utils.robot import BASE_CONFIG, ROBOT
from robot_ai.tasks import (
    Tool,
    analyze_if_tool_was_used_successfully,
    ask_for_help,
)


PICK_TEABAG_DROP_IN_CUP_DIR_PATH = (
    TOOL_EXECUTION_VIDEO_DIR_PATH / "pick_teabag_drop_in_cup"
)
REMOVE_TEABAG_FROM_CUP_DIR_PATH = (
    TOOL_EXECUTION_VIDEO_DIR_PATH / "remove_teabag_from_cup"
)
SUGAR_CUBE_IN_CUP_DIR_PATH = TOOL_EXECUTION_VIDEO_DIR_PATH / "sugar_cube_in_cup"
STIR_SPOON_DIR_PATH = TOOL_EXECUTION_VIDEO_DIR_PATH / "stir_spoon"
GENAI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
PICK_TEABAG_DROP_IN_CUP_SUCCESS_VALIDATION_QUESTION = (
    "Was the teacup stirred with the spoon that was inside the teacup?"
)
REMOVE_TEABAG_FROM_CUP_SUCCESS_VALIDATION_QUESTION = (
    "Was the teabag removed from the teacup?"
)
SUGAR_CUBE_IN_CUP_SUCCESS_VALIDATION_QUESTION = (
    "Was a sugar cube added into the teacup?"
)
STIR_SPOON_SUCCESS_VALIDATION_QUESTION = (
    "Was the teacup stirred with the spoon that was inside the teacup?"
)
PICK_TEABAG_DROP_IN_CUP_MODEL_PATH = os.environ.get("PICK_TEABAG_DROP_IN_CUP")
REMOVE_TEABAG_FROM_CUP_MODEL_PATH = os.environ.get("REMOVE_TEABAG_FROM_CUP")
SUGAR_CUBE_IN_CUP_MODEL_PATH = os.environ.get("SUGAR_CUBE_IN_CUP")
STIR_SPOON_MODEL_PATH = os.environ.get("STIR_SPOON")


def get_tea_making_tools() -> List[Tool]:
    return [
        Tool(
            name="pick_teabag_drop_in_cup",
            model_path=PICK_TEABAG_DROP_IN_CUP_MODEL_PATH,
            execution_time_in_seconds=40,
            success_validation_question=PICK_TEABAG_DROP_IN_CUP_SUCCESS_VALIDATION_QUESTION,
        ),
        Tool(
            name="remove_teabag_from_cup",
            model_path=REMOVE_TEABAG_FROM_CUP_MODEL_PATH,
            execution_time_in_seconds=40,
            success_validation_question=REMOVE_TEABAG_FROM_CUP_SUCCESS_VALIDATION_QUESTION,
        ),
        Tool(
            name="sugar_cube_in_cup",
            model_path=SUGAR_CUBE_IN_CUP_MODEL_PATH,
            execution_time_in_seconds=60,
            success_validation_question=SUGAR_CUBE_IN_CUP_SUCCESS_VALIDATION_QUESTION,
        ),
        Tool(
            name="stir_spoon",
            model_path=STIR_SPOON_MODEL_PATH,
            execution_time_in_seconds=40,
            success_validation_question=STIR_SPOON_SUCCESS_VALIDATION_QUESTION,
        ),
    ]


def remove_directory_if_exists(directory_path: Path):
    if os.path.isdir(directory_path):
        shutil.rmtree(directory_path, ignore_errors=True)


def pick_teabag_drop_in_cup():
    config = deepcopy(BASE_CONFIG)
    config.policy.pretrained_path = PICK_TEABAG_DROP_IN_CUP_MODEL_PATH
    config.dataset.episode_time_s = 40
    config.dataset.root = PICK_TEABAG_DROP_IN_CUP_DIR_PATH
    remove_directory_if_exists(config.dataset.root)
    record(cfg=config, robot=ROBOT)


def is_pick_teabag_drop_in_cup_success() -> bool:
    is_tool_usage_success = analyze_if_tool_was_used_successfully(
        ai_client=GENAI_CLIENT,
        video_file_name=PICK_TEABAG_DROP_IN_CUP_DIR_PATH.joinpath(RECORDED_VIDEO_PATH),
        question=PICK_TEABAG_DROP_IN_CUP_SUCCESS_VALIDATION_QUESTION,
    )
    return is_tool_usage_success


def remove_teabag_from_cup():
    config = deepcopy(BASE_CONFIG)
    config.policy.pretrained_path = REMOVE_TEABAG_FROM_CUP_MODEL_PATH
    config.dataset.episode_time_s = 40
    config.dataset.root = REMOVE_TEABAG_FROM_CUP_DIR_PATH
    remove_directory_if_exists(config.dataset.root)
    record(cfg=config, robot=ROBOT)


def is_remove_teabag_from_cup_success() -> bool:
    is_tool_usage_success = analyze_if_tool_was_used_successfully(
        ai_client=GENAI_CLIENT,
        video_file_name=REMOVE_TEABAG_FROM_CUP_DIR_PATH.joinpath(RECORDED_VIDEO_PATH),
        question=REMOVE_TEABAG_FROM_CUP_SUCCESS_VALIDATION_QUESTION,
    )
    return is_tool_usage_success


def sugar_cube_in_cup():
    config = deepcopy(BASE_CONFIG)
    config.policy.pretrained_path = SUGAR_CUBE_IN_CUP_MODEL_PATH
    config.dataset.episode_time_s = 60
    config.dataset.root = SUGAR_CUBE_IN_CUP_DIR_PATH
    remove_directory_if_exists(config.dataset.root)
    record(cfg=config, robot=ROBOT)


def is_sugar_cube_in_cup_success() -> bool:
    is_tool_usage_success = analyze_if_tool_was_used_successfully(
        ai_client=GENAI_CLIENT,
        video_file_name=SUGAR_CUBE_IN_CUP_DIR_PATH.joinpath(RECORDED_VIDEO_PATH),
        question=SUGAR_CUBE_IN_CUP_SUCCESS_VALIDATION_QUESTION,
    )
    return is_tool_usage_success


def stir_spoon():
    config = deepcopy(BASE_CONFIG)
    config.policy.pretrained_path = STIR_SPOON_MODEL_PATH
    config.dataset.episode_time_s = 40
    config.dataset.root = STIR_SPOON_DIR_PATH
    remove_directory_if_exists(config.dataset.root)
    record(cfg=config, robot=ROBOT)


def is_stir_spoon_success() -> bool:
    is_tool_usage_success = analyze_if_tool_was_used_successfully(
        ai_client=GENAI_CLIENT,
        video_file_name=STIR_SPOON_DIR_PATH.joinpath(RECORDED_VIDEO_PATH),
        question=STIR_SPOON_SUCCESS_VALIDATION_QUESTION,
    )
    return is_tool_usage_success


def ask_for_help_from_the_user():
    ask_for_help(ai_client=GENAI_CLIENT)
    input("Help the robot and finish its task. Press Enter to continue...")


def return_to_home():
    return_to_home_position(robot=ROBOT, home_position=HOME_POSITION)


@function_tool
def pick_teabag_drop_in_cup_function_tool():
    """Pick a tea bag and drop it into the teacup."""
    pick_teabag_drop_in_cup()


@function_tool
def is_pick_teabag_drop_in_cup_success_function_tool() -> bool:
    """Validate if the 'pick_teabag_drop_in_cup' tool usage was successful."""
    return is_pick_teabag_drop_in_cup_success()


@function_tool
def remove_teabag_from_cup_function_tool():
    """Remove the teabag from the teacup."""
    remove_teabag_from_cup()


@function_tool
def is_remove_teabag_from_cup_success_function_tool() -> bool:
    """Validate if the 'remove_teabag_from_cup' tool usage was successful."""
    return is_remove_teabag_from_cup_success()


@function_tool
def sugar_cube_in_cup_function_tool():
    """Add a sugar cube into the teacup."""
    sugar_cube_in_cup()


@function_tool
def is_sugar_cube_in_cup_success_function_tool() -> bool:
    """Validate if the 'sugar_cube_in_cup' tool usage was successful."""
    return is_sugar_cube_in_cup_success()


@function_tool
def stir_spoon_function_tool():
    """Stir the teacup with the spoon that is inside the teacup."""
    stir_spoon()


@function_tool
def is_stir_spoon_success_function_tool() -> bool:
    """Validate if the 'stir_spoon' tool usage was successful."""
    return is_stir_spoon_success()


@function_tool
def ask_for_help_from_the_user_function_tool():
    """Ask the user for help if the tool usage was not successful after 2 retries."""
    ask_for_help_from_the_user()


@function_tool
def return_to_home_function_tool():
    """Return the robot to its home position."""
    return_to_home()
