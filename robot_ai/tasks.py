import time
import wave
from typing import List
from dataclasses import dataclass

from google import genai
from google.genai import types
from pydub import AudioSegment
from pydub.playback import play

from robot_ai.constants import LLM_NAME, SCRIPT_DIR_PATH


@dataclass
class Tool:
    name: str
    model_path: str
    execution_time_in_seconds: int
    success_validation_question: str


def get_task_plan(
    ai_client: genai.Client, instruction: str, available_tools: List[Tool]
) -> List[Tool]:
    response = ai_client.models.generate_content(
        model=LLM_NAME,
        contents=(
            "List the tools in the right order so that the provided steps can be completed. "
            "Note that there might not be a tool for every step.\n\n"
            f"{instruction}\n"
            f"Available tools: {[tool.name for tool in available_tools]}"
        ),
        config={
            "response_mime_type": "application/json",
            "response_schema": list[str],
        },
    )
    task_plan = [
        available_tool
        for tool_name in response.parsed
        for available_tool in available_tools
        if available_tool.name == tool_name
    ]
    return task_plan


def analyze_if_tool_was_used_successfully(
    ai_client: genai.Client,
    video_file_name: str,
    question: str,
) -> bool:
    video_file = ai_client.files.upload(file=video_file_name)
    while not video_file.state.name == "ACTIVE":
        time.sleep(1)
        video_file = ai_client.files.get(name=video_file.name)
    response = ai_client.models.generate_content(
        model=LLM_NAME,
        contents=[
            video_file,
            question + " Please answer with yes or no.",
        ],
    )
    if "yes" in response.text.lower():
        return True
    return False


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def ask_for_help(ai_client: genai.Client):
    response = ai_client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents="Say in a calm way: I need help!",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Zephyr",
                    )
                )
            ),
        ),
    )

    data = response.candidates[0].content.parts[0].inline_data.data

    audio_file_name = SCRIPT_DIR_PATH / "ask_for_help.wav"
    wave_file(str(audio_file_name), data)
    song = AudioSegment.from_file(audio_file_name, format="wav")
    play(song)
