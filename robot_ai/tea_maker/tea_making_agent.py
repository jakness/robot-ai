import asyncio

from agents import Agent, Runner

from robot_ai.constants import LLM_NAME
from robot_ai.tea_maker.instructions import TEA_MAKING_INSTRUCTION_FOR_AGENT
from robot_ai.tea_maker.tools import (
    pick_teabag_drop_in_cup_function_tool,
    is_pick_teabag_drop_in_cup_success_function_tool,
    remove_teabag_from_cup_function_tool,
    is_remove_teabag_from_cup_success_function_tool,
    sugar_cube_in_cup_function_tool,
    is_sugar_cube_in_cup_success_function_tool,
    stir_spoon_function_tool,
    is_stir_spoon_success_function_tool,
    return_to_home_function_tool,
    ask_for_help_from_the_user_function_tool,
)
from robot_ai.lerobot_utils.robot import ROBOT


async def main():
    ROBOT.connect()
    agent = Agent(
        name="Tea maker",
        model=f"litellm/gemini/{LLM_NAME}",
        instructions=TEA_MAKING_INSTRUCTION_FOR_AGENT,
        tools=[
            pick_teabag_drop_in_cup_function_tool,
            is_pick_teabag_drop_in_cup_success_function_tool,
            remove_teabag_from_cup_function_tool,
            is_remove_teabag_from_cup_success_function_tool,
            sugar_cube_in_cup_function_tool,
            is_sugar_cube_in_cup_success_function_tool,
            stir_spoon_function_tool,
            is_stir_spoon_success_function_tool,
            return_to_home_function_tool,
            ask_for_help_from_the_user_function_tool,
        ],
    )

    result = await Runner.run(agent, "Make a cup of tea for me", max_turns=100)
    print(result.final_output)
    ROBOT.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
