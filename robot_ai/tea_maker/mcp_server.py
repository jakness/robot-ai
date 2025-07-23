from mcp.server.fastmcp import FastMCP

from robot_ai.tea_maker.tools import (
    pick_teabag_drop_in_cup,
    is_pick_teabag_drop_in_cup_success,
    remove_teabag_from_cup,
    is_remove_teabag_from_cup_success,
    sugar_cube_in_cup,
    is_sugar_cube_in_cup_success,
    stir_spoon,
    is_stir_spoon_success,
    return_to_home,
    ask_for_help_from_the_user,
)
from robot_ai.lerobot_utils.robot import ROBOT


mcp = FastMCP(name="RobotArmTools")


@mcp.tool()
def pick_teabag_drop_in_cup_tool():
    """Pick a tea bag and drop it into the teacup."""
    pick_teabag_drop_in_cup()


@mcp.tool()
def is_pick_teabag_drop_in_cup_success_tool() -> bool:
    """Validate if the 'pick_teabag_drop_in_cup' tool usage was successful."""
    return is_pick_teabag_drop_in_cup_success()


@mcp.tool()
def remove_teabag_from_cup_tool():
    """Remove the teabag from the teacup."""
    remove_teabag_from_cup()


@mcp.tool()
def is_remove_teabag_from_cup_success_tool() -> bool:
    """Validate if the 'remove_teabag_from_cup' tool usage was successful."""
    return is_remove_teabag_from_cup_success()


@mcp.tool()
def sugar_cube_in_cup_tool():
    """Add a sugar cube into the teacup."""
    sugar_cube_in_cup()


@mcp.tool()
def is_sugar_cube_in_cup_success_tool() -> bool:
    """Validate if the 'sugar_cube_in_cup' tool usage was successful."""
    return is_sugar_cube_in_cup_success()


@mcp.tool()
def stir_spoon_tool():
    """Stir the teacup with the spoon that is inside the teacup."""
    stir_spoon()


@mcp.tool()
def is_stir_spoon_success_tool() -> bool:
    """Validate if the 'stir_spoon' tool usage was successful."""
    return is_stir_spoon_success()


@mcp.tool()
def ask_for_help_from_the_user_tool():
    """Ask the user for help if the tool usage was not successful after 2 retries."""
    ask_for_help_from_the_user()


@mcp.tool()
def return_to_home_tool():
    """Return the robot to its home position."""
    return_to_home()


if __name__ == "__main__":
    ROBOT.connect()
    print("🚀 Starting Robot Arm MCP Server with tea maker tools...")
    mcp.run(transport="sse")
