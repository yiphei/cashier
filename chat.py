import argparse
import json
import os
import tempfile
from distutils.util import strtobool
from types import GeneratorType

from colorama import Fore, Style
from dotenv import load_dotenv  # Add this import
from elevenlabs import ElevenLabs, Voice, VoiceSettings, stream
from pydantic import BaseModel

from audio import get_audio_input, save_audio_to_wav
from chain import (
    FROM_NODE_ID_TO_EDGE_SCHEMA,
    confirm_order_node_schema,
    take_order_node_schema,
    terminal_order_node_schema,
)
from db_functions import create_db_client
from gui import remove_previous_line
from logger import logger
from model import CustomJSONEncoder, Model, ModelProvider, TurnContainer
from model_tool_decorator import FN_NAME_TO_FN

# Load environment variables from .env file
load_dotenv()


def get_text_from_speech(audio_data, oai_client):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
        # Save the audio data as a WAV file
        save_audio_to_wav(audio_data, temp_wav_file.name)

        # Use OpenAI's API to transcribe the saved WAV file
        transcription = oai_client.audio.transcriptions.create(
            model="whisper-1",
            language="en",
            file=open(temp_wav_file.name, "rb"),  # Open the saved WAV file for reading
        )
    return transcription.text


def get_speech_from_text(text_iterator, elabs_client):
    audio = elabs_client.generate(
        voice=Voice(
            voice_id="cgSgspJ2msm6clMCkdW9",
            settings=VoiceSettings(
                stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
            ),
        ),
        text=text_iterator,
        model="eleven_multilingual_v2",
        stream=True,
        optimize_streaming_latency=3,
    )
    stream(audio)


def get_user_input(use_audio_input, openai_client):
    if use_audio_input:
        audio_input = get_audio_input()
        text_input = get_text_from_speech(audio_input, openai_client)
    else:
        text_input = input("You: ")
        remove_previous_line()

    return text_input


class MessageDisplay:
    API_ROLE_TO_PREFIX = {
        "system": "System",
        "user": "You",
        "assistant": "Assistant",
    }

    API_ROLE_TO_COLOR = {
        "system": Fore.GREEN,
        "user": Fore.WHITE,
        "assistant": Fore.BLUE,
    }

    @classmethod
    def display_assistant_message(cls, message_or_stream):
        if isinstance(message_or_stream, GeneratorType):
            cls.print_msg(role="assistant", msg=None, end="")
            full_msg = ""
            for msg_chunk in message_or_stream:
                cls.print_msg(
                    role="assistant", msg=msg_chunk, add_role_prefix=False, end=""
                )
                full_msg += msg_chunk

            print("\n\n")
        else:
            cls.print_msg("assistant", message_or_stream)

    @classmethod
    def get_role_prefix(cls, role):
        return f"{Style.BRIGHT}{cls.API_ROLE_TO_PREFIX[role]}: {Style.NORMAL}"

    @classmethod
    def print_msg(cls, role, msg, add_role_prefix=True, end="\n\n"):
        formatted_msg = f"{cls.API_ROLE_TO_COLOR[role]}"
        if add_role_prefix:
            formatted_msg += f"{cls.get_role_prefix(role)}"
        if msg is not None:
            formatted_msg += f"{msg}"

        formatted_msg += f"{Style.RESET_ALL}"
        print(
            formatted_msg,
            end=end,
        )


def is_on_topic(model, TM, current_node_schema, all_node_schemas):
    model_name = "claude-3.5"
    model_provider = Model.get_model_provider(model_name)
    conversational_msgs = TM.model_provider_to_message_manager[
        model_provider
    ].conversation_dicts
    system_prompt = (
        "You are an AI-agent orchestration engine. Each AI agent is defined by an expectation"
        " and a set of tools (i.e. functions). Given the prior conversation, determine if the"
        " last user message can be fully handled by the current AI agent. Return true if"
        " the last user message is a case covered by the current AI agent's expectation OR "
        "tools. Return false if otherwise, meaning that we should explore letting another AI agent take over.\n\n"
        "LAST USER MESSAGE:\n"
        "```\n"
        f"{conversational_msgs[-1]['content']}\n"
        "```\n\n"
        "EXPECTATION:\n"
        "```\n"
        f"{current_node_schema.node_prompt}\n"
        "```\n\n"
        "TOOLS:\n"
        "```\n"
        f"{json.dumps(Model.get_tool_defs_from_names(current_node_schema.tool_fn_names, model_provider, current_node_schema.model_provider_to_tool_def[model_provider]))}\n"
        "```"
    )

    class Response1(BaseModel):
        output: bool

    model_provider = ModelProvider.ANTHROPIC
    chat_completion = model.chat(
        model_name=model_name,
        message_dicts=conversational_msgs,
        system=system_prompt,
        response_format=Response1,
        logprobs=True,
        temperature=0,
    )
    is_on_topic = chat_completion.get_message_prop("output")
    if model_provider == ModelProvider.OPENAI:
        prob = chat_completion.get_prob(-2)
        logger.debug(f"IS_ON_TOPIC: {is_on_topic} with {prob}")
    else:
        logger.debug(f"IS_ON_TOPIC: {is_on_topic}")
    if not is_on_topic:
        system_prompt = (
            "You are an AI-agent orchestration engine. Each AI agent is defined by an expectation"
            " and a set of tools (i.e. functions). An AI agent can handle a user message if it is "
            "a case covered by the AI agent's expectation OR tools. "
            "Given the prior conversation and a list of AI agents,"
            " determine which agent can best handle the last user message. "
            "Respond by returning the AI agent ID.\n\n"
        )
        for node_schema in all_node_schemas:
            system_prompt += (
                f"## AGENT ID: {node_schema.id}\n\n"
                "EXPECTATION:\n"
                "```\n"
                f"{node_schema.node_prompt}\n"
                "```\n\n"
                "TOOLS:\n"
                "```\n"
                f"{json.dumps(Model.get_tool_defs_from_names(node_schema.tool_fn_names, model_provider, node_schema.model_provider_to_tool_def[model_provider]))}\n"
                "```\n\n"
            )

        system_prompt += (
            "LAST USER MESSAGE:\n"
            "```\n"
            f"{conversational_msgs[-1]['content']}\n"
            "```"
        )

        class Response2(BaseModel):
            agent_id: int

        chat_completion = model.chat(
            model_name=model_name,
            message_dicts=conversational_msgs,
            system=system_prompt,
            response_format=Response2,
            logprobs=True,
            temperature=0,
        )

        agent_id = chat_completion.get_message_prop("agent_id")
        if model_provider == ModelProvider.OPENAI:
            prob = chat_completion.get_prob(-2)
            logger.debug(f"AGENT_ID: {agent_id} with {prob}")
        else:
            logger.debug(f"AGENT_ID: {agent_id}")


def run_chat(args, model, elevenlabs_client):
    TC = TurnContainer()

    need_user_input = True
    current_node_schema = take_order_node_schema
    current_edge_schemas = FROM_NODE_ID_TO_EDGE_SCHEMA[current_node_schema.id]
    new_node_input = None

    while True:
        if not current_node_schema.is_initialized:
            logger.debug(
                f"[NODE_SCHEMA] Initializing {Style.BRIGHT}node_schema_id: {current_node_schema.id}{Style.NORMAL}"
            )
            current_node_schema.run(new_node_input)
            TC.add_node_turn(
                current_node_schema.id,
                current_node_schema.prompt,
                remove_prev_tool_calls=True,
            )
            MessageDisplay.print_msg("system", current_node_schema.prompt)

            if current_node_schema.first_msg:
                # TODO fix this
                TC.add_assistant_turn(
                    current_node_schema.first_msg["content"], ModelProvider.NONE
                )
                MessageDisplay.print_msg(
                    "assistant", current_node_schema.first_msg["content"]
                )

        if need_user_input:
            # Read user input from stdin
            text_input = get_user_input(args.audio_input, model.oai_client)
            # If user types 'exit', break the loop and end the program
            if text_input.lower() == "exit":
                print("Exiting chatbot. Goodbye!")
                break
            MessageDisplay.print_msg("user", text_input)
            TC.add_user_turn(text_input)
            is_on_topic(
                model,
                TC,
                current_node_schema,
                [
                    take_order_node_schema,
                    confirm_order_node_schema,
                    terminal_order_node_schema,
                ],
            )

        chat_completion = model.chat(
            model_name=args.model,
            turn_container=TC,
            tool_names_or_tool_defs=current_node_schema.tool_fn_names,
            stream=args.stream,
            model_provider_to_extra_tool_defs=current_node_schema.model_provider_to_tool_def,
        )
        message = chat_completion.get_or_stream_message()
        if message is not None:
            if args.audio_output:
                get_speech_from_text(message, elevenlabs_client)
            else:
                MessageDisplay.display_assistant_message(message)
            need_user_input = True

        fn_id_to_output = {}
        for function_call in chat_completion.get_or_stream_fn_calls():
            function_args = function_call.function_args
            logger.debug(
                f"[FUNCTION_CALL] {Style.BRIGHT}name: {function_call.function_name}, id: {function_call.tool_call_id}{Style.NORMAL} with args:\n{json.dumps(function_args, indent=4)}"
            )

            if function_call.function_name.startswith("get_state"):
                fn_output = getattr(current_node_schema, function_call.function_name)(
                    **function_args
                )
            elif function_call.function_name.startswith("update_state"):
                fn_output = current_node_schema.update_state(**function_args)
                state_condition_results = [
                    edge_schema.check_state_condition(current_node_schema.state)
                    for edge_schema in current_edge_schemas
                ]
                if any(state_condition_results):
                    first_true_index = state_condition_results.index(True)
                    first_true_edge_schema = current_edge_schemas[first_true_index]

                    new_node_input = first_true_edge_schema.new_input_from_state_fn(
                        current_node_schema.state
                    )
                    current_node_schema = first_true_edge_schema.to_node_schema
                    current_edge_schemas = FROM_NODE_ID_TO_EDGE_SCHEMA.get(
                        current_node_schema.id, []
                    )
            else:
                fn = FN_NAME_TO_FN[function_call.function_name]
                fn_output = fn(**function_args)

            logger.debug(
                f"[FUNCTION_RETURN] {Style.BRIGHT}name: {function_call.function_name}, id: {function_call.tool_call_id}{Style.NORMAL} with output:\n{json.dumps(fn_output, cls=CustomJSONEncoder, indent=4)}"
            )
            need_user_input = False
            fn_id_to_output[function_call.tool_call_id] = fn_output

        model_provider = Model.get_model_provider(args.model)
        TC.add_assistant_turn(
            chat_completion.msg_content,
            model_provider,
            chat_completion.fn_calls,
            fn_id_to_output,
        )


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
    args = parser.parse_args()

    model = Model()
    elevenlabs_client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    create_db_client()

    if not args.enable_logging:
        logger.disabled = True
    run_chat(args, model, elevenlabs_client)
