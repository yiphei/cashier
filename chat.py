import argparse
import json
import os
import tempfile
from collections import defaultdict
from distutils.util import strtobool

import numpy as np
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
from db_functions import FN_NAME_TO_FN, OPENAI_TOOLS_RETUN_DESCRIPTION, create_db_client
from gui import remove_previous_line
from logger import logger
from model import Model

# Load environment variables from .env file
load_dotenv()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, (defaultdict, dict)):
            return {self.default(k): self.default(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        return super().default(obj)


def get_system_return_type_prompt(fn_name):
    json_schema = OPENAI_TOOLS_RETUN_DESCRIPTION[fn_name]
    return (
        f"This is the JSON Schema of {fn_name}'s return type: {json.dumps(json_schema)}"
    )


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


class ListIndexTracker:
    def __init__(self):
        self.named_idx_to_idx = {}
        self.idx_to_named_idx = {}
        self.idxs = []
        self.idx_to_pos = {}

    def add_idx(self, named_idx, idx):
        self.named_idx_to_idx[named_idx] = idx
        self.idx_to_named_idx[idx] = named_idx
        self.idxs.append(idx)
        self.idx_to_pos[idx] = len(self.idxs) - 1

    def get_idx(self, named_idx):
        return self.named_idx_to_idx[named_idx]

    def pop_idx(self, named_idx):
        popped_idx = self.named_idx_to_idx.pop(named_idx)
        popped_idx_pos = self.idx_to_pos.pop(popped_idx)
        self.idx_to_named_idx.pop(popped_idx)
        del self.idxs[popped_idx_pos]

        for i in range(popped_idx_pos, len(self.idxs)):
            curr_idx = self.idxs[i]
            curr_named_idx = self.idx_to_named_idx[curr_idx]

            self.idxs[i] -= 1
            self.idx_to_pos.pop(curr_idx)
            self.idx_to_pos[self.idxs[i]] = i

            self.named_idx_to_idx[curr_named_idx] = self.idxs[i]
            self.idx_to_named_idx.pop(curr_idx)
            self.idx_to_named_idx[self.idxs[i]] = curr_named_idx
        return popped_idx


class MessageManager:
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

    def __init__(self, output_system_prompt=False):
        self.messages = []
        self.output_system_prompt = output_system_prompt
        self.list_index_tracker = ListIndexTracker()
        self.last_node_id = None
        self.tool_call_ids = []
        self.tool_fn_return_names = set()

    def add_message_dict(self, msg_dict, print_msg=True):
        self.messages.append(msg_dict)
        if (
            print_msg
            and (msg_dict["role"] != "system" or self.output_system_prompt)
            and not self.is_tool_message(msg_dict)
        ):
            self.print_msg(msg_dict["role"], msg_dict["content"])

    def add_user_message(self, msg):
        self.add_message_dict({"role": "user", "content": msg})

    def add_assistant_message(self, msg):
        self.add_message_dict({"role": "assistant", "content": msg})

    def add_system_message(self, msg):
        self.add_message_dict({"role": "system", "content": msg})

    def add_tool_call_message(self, tool_call_id, function_name, function_args_json):
        self.add_message_dict(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "arguments": function_args_json,
                            "name": function_name,
                        },
                    }
                ],
            }
        )
        self.list_index_tracker.add_idx(tool_call_id, len(self.messages) - 1)
        self.tool_call_ids.append(tool_call_id)

    def add_tool_response_message(self, tool_call_id, tool_response):
        self.add_message_dict(
            {
                "role": "tool",
                "content": tool_response,
                "tool_call_id": tool_call_id,
            }
        )
        self.list_index_tracker.add_idx(tool_call_id + "return", len(self.messages) - 1)

    def add_tool_return_schema_message(self, tool_name, msg):
        if tool_name in self.list_index_tracker.named_idx_to_idx:
            idx_to_remove = self.list_index_tracker.pop_idx(tool_name)
            del self.messages[idx_to_remove]

        self.add_system_message(msg)
        self.list_index_tracker.add_idx(tool_name, len(self.messages) - 1)
        self.tool_fn_return_names.add(tool_name)

    def add_node_system_message(
        self,
        node_id,
        msg,
        remove_prev_tool_fn_return=None,
        remove_prev_tool_calls=False,
    ):
        if remove_prev_tool_calls:
            assert remove_prev_tool_fn_return is not False

        if self.last_node_id is not None:
            idx_to_remove = self.list_index_tracker.pop_idx(self.last_node_id)
            del self.messages[idx_to_remove]

        if remove_prev_tool_fn_return is True or remove_prev_tool_calls:
            for toll_return in self.tool_fn_return_names:
                idx_to_remove = self.list_index_tracker.pop_idx(toll_return)
                del self.messages[idx_to_remove]

            self.tool_fn_return_names = set()

        if remove_prev_tool_calls:
            for tool_call_id in self.tool_call_ids:
                idx_to_remove = self.list_index_tracker.pop_idx(tool_call_id)
                del self.messages[idx_to_remove]

                idx_to_remove = self.list_index_tracker.pop_idx(tool_call_id + "return")
                del self.messages[idx_to_remove]

            self.tool_call_ids = []

        self.add_system_message(msg)
        self.list_index_tracker.add_idx(node_id, len(self.messages) - 1)
        self.last_node_id = node_id

    def is_tool_message(self, msg):
        return (msg["role"] == "assistant" and msg.get("tool_calls") is not None) or (
            msg["role"] == "tool" and msg.get("tool_call_id") is not None
        )

    def read_chat_stream(self, chat_stream):
        self.print_msg(role="assistant", msg=None, end="")
        full_msg = ""
        for msg_chunk in chat_stream:
            self.print_msg(
                role="assistant", msg=msg_chunk, add_role_prefix=False, end=""
            )
            full_msg += msg_chunk

        self.add_message_dict(
            {"role": "assistant", "content": full_msg},
            print_msg=False,
        )
        print("\n\n")

    def get_role_prefix(self, role):
        return f"{Style.BRIGHT}{self.API_ROLE_TO_PREFIX[role]}: {Style.NORMAL}"

    def print_msg(self, role, msg, add_role_prefix=True, end="\n\n"):
        formatted_msg = f"{self.API_ROLE_TO_COLOR[role]}"
        if add_role_prefix:
            formatted_msg += f"{self.get_role_prefix(role)}"
        if msg is not None:
            formatted_msg += f"{msg}"

        formatted_msg += f"{Style.RESET_ALL}"
        print(
            formatted_msg,
            end=end,
        )

    def is_conversational_msg(self, msg):
        return msg["role"] == "user" or (
            msg["role"] == "assistant" and not self.is_tool_message(msg)
        )

    def get_all_conversational_messages_of_current_node(self):
        start_idx = self.list_index_tracker.get_idx(self.last_node_id)
        return [
            msg
            for msg in self.messages[start_idx + 1 :]
            if self.is_conversational_msg(msg)
        ]


def is_on_topic(model, MM, current_node_schema, all_node_schemas):
    conversational_msgs = MM.get_all_conversational_messages_of_current_node()
    conversational_msgs.append(
        {
            "role": "system",
            "content": (
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
                f"{json.dumps(current_node_schema.tool_fns)}\n"
                "```"
            ),
        }
    )

    class Response1(BaseModel):
        output: bool

    chat_completion = model.chat(
        model_name="gpt-4o-mini",
        messages=conversational_msgs,
        response_format=Response1,
        logprobs=True,
        temperature=0,
    )
    is_on_topic = chat_completion.output_obj.choices[0].message.parsed.output
    prob = np.exp(
        chat_completion.output_obj.choices[0].logprobs.content[-2].logprob
    )
    logger.debug(f"IS_ON_TOPIC: {is_on_topic} with {prob}")
    if not is_on_topic:
        conversational_msgs.pop()
        prompt = (
            "You are an AI-agent orchestration engine. Each AI agent is defined by an expectation"
            " and a set of tools (i.e. functions). An AI agent can handle a user message if it is "
            "a case covered by the AI agent's expectation OR tools. "
            "Given the prior conversation and a list of AI agents,"
            " determine which agent can best handle the last user message. "
            "Respond by returning the AI agent ID.\n\n"
        )
        for node_schema in all_node_schemas:
            prompt += (
                f"## AGENT ID: {node_schema.id}\n\n"
                "EXPECTATION:\n"
                "```\n"
                f"{node_schema.node_prompt}\n"
                "```\n\n"
                "TOOLS:\n"
                "```\n"
                f"{json.dumps(node_schema.tool_fns)}\n"
                "```\n\n"
            )

        prompt += (
            "LAST USER MESSAGE:\n"
            "```\n"
            f"{conversational_msgs[-1]['content']}\n"
            "```"
        )
        conversational_msgs.append({"role": "system", "content": prompt})

        class Response2(BaseModel):
            agent_id: int

        chat_completion = model.chat(
            model_name="gpt-4o",
            messages=conversational_msgs,
            response_format=Response2,
            logprobs=True,
            temperature=0,
        )

        agent_id = chat_completion.output_obj.choices[0].message.parsed.agent_id
        prob = np.exp(
            chat_completion.output_obj.choices[0].logprobs.content[-2].logprob
        )
        logger.debug(f"AGENT_ID: {agent_id} with {prob}")


def run_chat(args, model, elevenlabs_client):
    MM = MessageManager(
        output_system_prompt=args.output_system_prompt,
    )

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
            MM.add_node_system_message(
                current_node_schema.id, current_node_schema.prompt
            )
            if current_node_schema.first_msg:
                MM.add_message_dict(current_node_schema.first_msg)

        if need_user_input:
            # Read user input from stdin
            text_input = get_user_input(args.audio_input, model.oai_client)
            # If user types 'exit', break the loop and end the program
            if text_input.lower() == "exit":
                print("Exiting chatbot. Goodbye!")
                break

            MM.add_user_message(text_input)
            is_on_topic(
                model,
                MM,
                current_node_schema,
                [
                    take_order_node_schema,
                    confirm_order_node_schema,
                    terminal_order_node_schema,
                ],
            )

        chat_completion = model.chat(
            model_name=args.model,
            messages=MM.messages,
            tools=current_node_schema.tool_fns,
            stream=True,
        )

        has_tool_call = chat_completion.has_tool_call()

        if not has_tool_call:
            if args.audio_output:
                get_speech_from_text(chat_completion.iter_messages(), elevenlabs_client)
                MM.add_assistant_message(chat_completion.msg_content)
            else:
                MM.read_chat_stream(chat_completion.iter_messages())
            need_user_input = True
        elif has_tool_call:
            function_calls = chat_completion.extract_fn_calls()

            for function_call in function_calls:
                function_args = json.loads(function_call.function_args_json)
                logger.debug(
                    f"[FUNCTION_CALL] {Style.BRIGHT}name: {function_call.function_name}, id: {function_call.tool_call_id}{Style.NORMAL} with args:\n{json.dumps(function_args, indent=4)}"
                )

                if function_call.function_name.startswith("get_state"):
                    fn_output = getattr(
                        current_node_schema, function_call.function_name
                    )(**function_args)
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
                MM.add_tool_call_message(
                    function_call.tool_call_id,
                    function_call.function_name,
                    function_call.function_args_json,
                )
                MM.add_tool_response_message(
                    function_call.tool_call_id,
                    json.dumps(fn_output, cls=CustomJSONEncoder),
                )

                if not function_call.function_name.startswith(
                    ("get_state", "update_state")
                ):
                    MM.add_tool_return_schema_message(
                        function_call.function_name,
                        get_system_return_type_prompt(function_call.function_name),
                    )

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
    args = parser.parse_args()

    model = Model()
    elevenlabs_client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    create_db_client()

    if not args.enable_logging:
        logger.disabled = True
    run_chat(args, model, elevenlabs_client)
