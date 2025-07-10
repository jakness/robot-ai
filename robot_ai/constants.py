import os
from pathlib import Path


SCRIPT_DIR_PATH = Path(__file__).resolve().parent
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TOOL_EXECUTION_VIDEO_DIR_PATH = SCRIPT_DIR_PATH.parent / "recorded_tool_executions"
LLM_NAME = "gemini-2.5-pro"
