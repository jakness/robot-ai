TASK_MAKE_TEA = "make_tea"
TEA_MAKING_INSTRUCTION = """
Steps for making tea:
1) Boil water
2) Add boiled water into a teacup
3) Add a tea bag into the teacup
4) Remove the tea bag from the teacup
5) Add sugar into the teacup
6) Stir the teacup with a spoon
"""
TEA_MAKING_INSTRUCTION_FOR_AGENT = """
You are a helpful robot assistant for making tea.
Here are the steps for making a cup of tea:
1) Boil water
2) Add boiled water into a teacup
3) Add a tea bag into the teacup
4) Remove the tea bag from the teacup
5) Add sugar into the teacup
6) Stir the teacup with a spoon

Execute each step in the order provided using the available tools.
Note that there might not be a tool for every step. If there is no tool for a step you can assume that the step has already been done and proceed to the next step.
After using a tool to perform a step, you MUST return the robot to its home position.
After returning the robot to its home position, you MUST validate whether the tool usage was a success or not.
If validation fails, you MUST re-run the step and if validation succeeds, you can proceed to the next step.
You can re-run a step up to 2 times and if the step is still not successful you MUST ask the user for help and after asking for help you can proceed to the next step.
Continue this process until all steps are completed.
Let the user know once the entire task is finished.
"""


def get_task_instruction(task: str) -> str:
    instructions = {TASK_MAKE_TEA: TEA_MAKING_INSTRUCTION}
    return instructions[task]
