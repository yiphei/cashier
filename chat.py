import argparse
import os
from argparse import Namespace
from distutils.util import strtobool
from typing import Any

from dotenv import load_dotenv  # Add this import
from elevenlabs import ElevenLabs

from cashier.agent_executor import AgentExecutor
from cashier.audio import get_audio_input, get_text_from_speech
from cashier.graph_data.cashier import cashier_graph_schema
from cashier.gui import remove_previous_line
from cashier.logger import logger
from cashier.model import Model
from cashier.tool_registries.cashier_tool_registry import create_db_client

# Load environment variables from .env file
load_dotenv()


def get_user_input(use_audio_input: bool, openai_client: Any) -> str:
    if use_audio_input:
        audio_input = get_audio_input()
        text_input = get_text_from_speech(audio_input, openai_client)
    else:
        text_input = input("You: ")
        remove_previous_line()

    return text_input


def run_chat(args: Namespace, model: Model, elevenlabs_client: Any):
    model_provider = Model.get_model_provider(args.model)
    AE = AgentExecutor(
        model,
        elevenlabs_client,
        cashier_graph_schema,
        args.audio_output,
        args.remove_prev_tool_calls,
        model_provider,
    )

    while True:
        if AE.need_user_input:
            # Read user input from stdin
            text_input = get_user_input(args.audio_input, model.oai_client)
            # If user types 'exit', break the loop and end the program
            if text_input.lower() == "exit":
                print("Exiting chatbot. Goodbye!")
                break
            AE.add_user_turn(text_input)

        chat_completion = model.chat(
            model_name=args.model,
            stream=args.stream,
            **AE.get_model_completion_kwargs(),
        )
        AE.add_assistant_turn(chat_completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_input",
        type=lambda v: bool(strtobool(v)),
        default=False,
    )
    parser.add_argument(
        "--audio_output",
        type=lambda v: bool(strtobool(v)),
        default=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--output_system_prompt",
        type=lambda v: bool(strtobool(v)),
        default=True,
    )
    parser.add_argument(
        "--enable_logging",
        type=lambda v: bool(strtobool(v)),
        default=True,
    )
    parser.add_argument(
        "--stream",
        type=lambda v: bool(strtobool(v)),
        default=True,
    )
    parser.add_argument(
        "--remove_prev_tool_calls",
        type=lambda v: bool(strtobool(v)),
        default=False,
    )
    args = parser.parse_args()

    model = Model()
    elevenlabs_client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    create_db_client()

    if not args.enable_logging:
        logger.disabled = True
    run_chat(args, model, elevenlabs_client)
