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


def get_task_instruction(task: str) -> str:
    instructions = {TASK_MAKE_TEA: TEA_MAKING_INSTRUCTION}
    return instructions[task]
