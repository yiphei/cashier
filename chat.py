import argparse
import json
import os
import tempfile
from distutils.util import strtobool
from types import GeneratorType

from colorama import Fore, Style
from dotenv import load_dotenv  # Add this import
from elevenlabs import ElevenLabs, Voice, VoiceSettings, stream

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
from model import (
    AssistantModelTurn,
    CustomJSONEncoder,
    Model,
    NodeSystemTurn,
    TurnContainer,
    UserTurn,
    ModelProvider
)
from model_tool_decorator import FN_NAME_TO_FN, OPENAI_TOOL_NAME_TO_TOOL_DEF

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
    def read_chat_stream(cls, chat_stream):
        if isinstance(chat_stream, GeneratorType):
            cls.print_msg(role="assistant", msg=None, end="")
            full_msg = ""
            for msg_chunk in chat_stream:
                cls.print_msg(
                    role="assistant", msg=msg_chunk, add_role_prefix=False, end=""
                )
                full_msg += msg_chunk

            print("\n\n")
        else:
            cls.print_msg("assistant", chat_stream)

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
    all_tool_defs = (
        OPENAI_TOOL_NAME_TO_TOOL_DEF | current_node_schema.OPENAI_TOOL_NAME_TO_TOOL_DEF
    )
    # conversational_msgs = TM.get_all_conversational_messages_of_current_node()
    # conversational_msgs.append(
    #     {
    #         "role": "system",
    #         "content": (
    #             "You are an AI-agent orchestration engine. Each AI agent is defined by an expectation"
    #             " and a set of tools (i.e. functions). Given the prior conversation, determine if the"
    #             " last user message can be fully handled by the current AI agent. Return true if"
    #             " the last user message is a case covered by the current AI agent's expectation OR "
    #             "tools. Return false if otherwise, meaning that we should explore letting another AI agent take over.\n\n"
    #             "LAST USER MESSAGE:\n"
    #             "```\n"
    #             f"{conversational_msgs[-1]['content']}\n"
    #             "```\n\n"
    #             "EXPECTATION:\n"
    #             "```\n"
    #             f"{current_node_schema.node_prompt}\n"
    #             "```\n\n"
    #             "TOOLS:\n"
    #             "```\n"
    #             f"{json.dumps([all_tool_defs[name] for name in current_node_schema.tool_fn_names])}\n"
    #             "```"
    #         ),
    #     }
    # )

    # class Response1(BaseModel):
    #     output: bool

    # chat_completion = model.chat(
    #     model_name="gpt-4o-mini",
    #     messages=conversational_msgs,
    #     response_format=Response1,
    #     logprobs=True,
    #     temperature=0,
    # )
    # is_on_topic = chat_completion.get_message_prop("output")
    # prob = chat_completion.get_prob(-2)
    # logger.debug(f"IS_ON_TOPIC: {is_on_topic} with {prob}")
    # if not is_on_topic:
    #     conversational_msgs.pop()
    #     prompt = (
    #         "You are an AI-agent orchestration engine. Each AI agent is defined by an expectation"
    #         " and a set of tools (i.e. functions). An AI agent can handle a user message if it is "
    #         "a case covered by the AI agent's expectation OR tools. "
    #         "Given the prior conversation and a list of AI agents,"
    #         " determine which agent can best handle the last user message. "
    #         "Respond by returning the AI agent ID.\n\n"
    #     )
    #     for node_schema in all_node_schemas:
    #         prompt += (
    #             f"## AGENT ID: {node_schema.id}\n\n"
    #             "EXPECTATION:\n"
    #             "```\n"
    #             f"{node_schema.node_prompt}\n"
    #             "```\n\n"
    #             "TOOLS:\n"
    #             "```\n"
    #             f"{json.dumps([all_tool_defs[name] for name in current_node_schema.tool_fn_names])}\n"
    #             "```\n\n"
    #         )

    #     prompt += (
    #         "LAST USER MESSAGE:\n"
    #         "```\n"
    #         f"{conversational_msgs[-1]['content']}\n"
    #         "```"
    #     )
    #     conversational_msgs.append({"role": "system", "content": prompt})

    #     class Response2(BaseModel):
    #         agent_id: int

    #     chat_completion = model.chat(
    #         model_name="gpt-4o",
    #         messages=conversational_msgs,
    #         response_format=Response2,
    #         logprobs=True,
    #         temperature=0,
    #     )

    #     agent_id = chat_completion.get_message_prop("agent_id")
    #     prob = chat_completion.get_prob(-2)
    #     logger.debug(f"AGENT_ID: {agent_id} with {prob}")


def run_chat(args, model, elevenlabs_client):
    TM = TurnContainer()

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
            node_system_turn = NodeSystemTurn(
                node_id=current_node_schema.id, msg_content=current_node_schema.prompt
            )
            TM.add_node_turn(node_system_turn)
            MessageDisplay.print_msg("system", node_system_turn.msg_content)

            if current_node_schema.first_msg:
                # TODO fix this
                a_turn = AssistantModelTurn(
                    msg_content=current_node_schema.first_msg["content"]
                )
                TM.add_assistant_turn(a_turn)
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
            user_turn = UserTurn(msg_content=text_input)
            MessageDisplay.print_msg("user", text_input)
            TM.add_user_turn(user_turn)
            is_on_topic(
                model,
                TM,
                current_node_schema,
                [
                    take_order_node_schema,
                    confirm_order_node_schema,
                    terminal_order_node_schema,
                ],
            )

        chat_completion = model.chat(
            model_name=args.model,
            messages=TM.get_message_dicts(ModelProvider.ANTHROPIC if args.model == 'claude-3.5' else ModelProvider.OPENAI),
            tool_names=current_node_schema.tool_fn_names,
            stream=args.stream,
            extra_oai_tool_defs=current_node_schema.OPENAI_TOOL_NAME_TO_TOOL_DEF,
            extra_anthropic_tool_defs=current_node_schema.ANTHROPIC_TOOL_NAME_TO_TOOL_DEF,
        )
        message = chat_completion.get_or_stream_message()
        if message is not None:
            if args.audio_output:
                get_speech_from_text(message, elevenlabs_client)
            else:
                MessageDisplay.read_chat_stream(message)
            need_user_input = True

        a_turn_2 = AssistantModelTurn(msg_content=chat_completion.msg_content)

        for function_call in chat_completion.get_or_stream_fn_calls():
            function_args = json.loads(function_call.function_args_json)
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
            a_turn_2.add_fn_call_w_output(function_call, fn_output)
            TM.add_assistant_turn(a_turn_2)
            need_user_input = False


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
