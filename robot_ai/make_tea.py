import os
import logging
import sys
from typing import List

from google import genai
from lerobot.record import RecordConfig
from lerobot.configs import parser
from lerobot.common.robots.utils import make_robot_from_config

from robot_ai.constants import GEMINI_API_KEY, TOOL_EXECUTION_VIDEO_DIR_PATH
from robot_ai.lerobot_utils.record import (
    record,
    return_to_home_position,
    HOME_POSITION,
    RECORDED_VIDEO_PATH,
)
from robot_ai.instructions import get_task_instruction, TASK_MAKE_TEA
from robot_ai.tasks import (
    Tool,
    get_task_plan,
    analyze_if_tool_was_used_successfully,
    ask_for_help,
)


logger = logging.getLogger()


def get_tea_making_tools() -> List[Tool]:
    return [
        Tool(
            name="pick_teabag_drop_in_cup",
            model_path=os.environ.get("PICK_TEABAG_DROP_IN_CUP"),
            execution_time_in_seconds=40,
            success_validation_question="Was a teabag added into the teacup?",
        ),
        Tool(
            name="remove_teabag_from_cup",
            model_path=os.environ.get("REMOVE_TEABAG_FROM_CUP"),
            execution_time_in_seconds=40,
            success_validation_question="Was the teabag removed from the teacup?",
        ),
        Tool(
            name="sugar_cube_in_cup",
            model_path=os.environ.get("SUGAR_CUBE_IN_CUP"),
            execution_time_in_seconds=60,
            success_validation_question="Was a sugar cube added into the teacup?",
        ),
        Tool(
            name="stir_spoon",
            model_path=os.environ.get("STIR_SPOON"),
            execution_time_in_seconds=40,
            success_validation_question="Was the teacup stirred with the spoon that was inside the teacup?",
        ),
    ]


@parser.wrap()
def main(cfg: RecordConfig):
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    tea_making_tools = get_tea_making_tools()
    task_instruction = get_task_instruction(task=TASK_MAKE_TEA)
    task_plan = get_task_plan(
        ai_client=genai_client,
        instruction=task_instruction,
        available_tools=tea_making_tools,
    )

    # Sanity check. Can't be used in production
    assert task_plan == tea_making_tools, (
        "The tools used are wrong or in the wrong order for making tea"
    )

    for i, tool in enumerate(task_plan):
        is_tool_usage_success = False
        tool_execution_tries = 0

        cfg.policy.pretrained_path = tool.model_path
        cfg.dataset.episode_time_s = tool.execution_time_in_seconds
        while not is_tool_usage_success:
            tool_execution_video_dir_path = (
                TOOL_EXECUTION_VIDEO_DIR_PATH / f"{tool.name}_{tool_execution_tries}"
            )
            cfg.dataset.root = tool_execution_video_dir_path

            logger.info(f"Running tool: {tool.name} (try {tool_execution_tries + 1})")
            record(cfg=cfg, robot=robot)

            logger.info("Return to home position")
            return_to_home_position(robot=robot, home_position=HOME_POSITION)

            logger.info("Analyzing if the tool was used successfully...")
            is_tool_usage_success = analyze_if_tool_was_used_successfully(
                ai_client=genai_client,
                video_file_name=tool_execution_video_dir_path.joinpath(
                    RECORDED_VIDEO_PATH
                ),
                question=tool.success_validation_question,
            )

            tool_execution_tries += 1
            if tool_execution_tries >= 3 and not is_tool_usage_success:
                ask_for_help(ai_client=genai_client)
                input(
                    f"Robot had trouble using the skill {tool.name}. "
                    "Help the robot by executing the skill yourself and press Enter to continue..."
                )
                is_tool_usage_success = True

    robot.disconnect()


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            # redirect the log output to the same stream used by input() so that the messages are displayed in chronological order
            handler.stream = sys.stdout
    main()
