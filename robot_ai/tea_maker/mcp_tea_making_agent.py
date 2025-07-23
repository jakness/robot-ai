import asyncio
import time

from agents import Agent, Runner
from agents.mcp import MCPServerSse

from robot_ai.constants import LLM_NAME
from robot_ai.tea_maker.instructions import TEA_MAKING_INSTRUCTION_FOR_AGENT


async def main():
    mcp_server = MCPServerSse(
        params={"url": "http://localhost:8000/sse"},
        cache_tools_list=True,
        client_session_timeout_seconds=3000,
    )

    async with mcp_server:
        time.sleep(5)
        agent = Agent(
            name="Tea maker (MCP)",
            model=f"litellm/gemini/{LLM_NAME}",
            instructions=TEA_MAKING_INSTRUCTION_FOR_AGENT,
            mcp_servers=[mcp_server],
        )

        result = await Runner.run(agent, "Make a cup of tea for me", max_turns=100)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
