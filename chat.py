import argparse
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from distutils.util import strtobool

from dotenv import load_dotenv  # Add this import
from elevenlabs import ElevenLabs, Voice, VoiceSettings, stream
from openai import OpenAI
from pydantic import BaseModel

from audio import get_audio_input, save_audio_to_wav
from chain import FROM_NODE_ID_TO_EDGE_SCHEMA, take_order_node_schema
from db_functions import FN_NAME_TO_FN, OPENAI_TOOLS_RETUN_DESCRIPTION, create_db_client

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = (
    "You are a cashier working for the coffee shop Heaven Coffee, and you are physically embedded in it, "
    "meaning you will interact with real in-person customers. There is a microphone that transcribes customer's speech to text, "
    "and a speaker that outputs your text to speech. Because your responses will be converted to speech, "
    "you must respond in a conversational way: natural and easy to understand when converted to speech. So do not use "
    "any text formatting like hashtags, bold, italic, bullet points, etc.\n\n"
    "Because you are conversing, keep your responses generally concise and brief, and "
    "do not provide unrequested information. "
    "If a response to a request is naturally long, then either ask claryfing questions to further refine the request, "
    "or break down the response in many separate responses.\n\n"
    "Overall, be professional, polite, empathetic, and friendly.\n\n"
    "Lastly, minimize reliance on external knowledge. Always get information from the prompts and tools."
    "If they dont provide the information you need, just say you do not know."
)


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


class ChatCompletionIterator(Iterator):
    def __init__(self, chat_stream):
        self.chat_stream = chat_stream
        self.full_msg = ""  # Initialize full message

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.chat_stream)  # Get the next chunk
            msg = chunk.choices[0].delta.content
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason is not None:
                raise StopIteration
            if msg is None:
                print("WARNING: msg is None")
                print(chunk)
                raise StopIteration
            self.full_msg += msg  # Append the message to full_msg
            return msg  # Return the message
        except StopIteration:
            raise StopIteration  # Signal end of iteration


def get_system_return_type_prompt(fn_name):
    json_schema = OPENAI_TOOLS_RETUN_DESCRIPTION[fn_name]
    msg = {
        "role": "system",
        "content": f"This is the JSON Schema of {fn_name}'s return type: {json.dumps(json_schema)}",
    }
    return msg


def get_text_from_speech(audio_data, client):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
        # Save the audio data as a WAV file
        save_audio_to_wav(audio_data, temp_wav_file.name)

        # Use OpenAI's API to transcribe the saved WAV file
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            language="en",
            file=open(temp_wav_file.name, "rb"),  # Open the saved WAV file for reading
        )
    return transcription.text


def get_speech_from_text(text_iterator, client):
    audio = client.generate(
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


class FunctionCall(BaseModel):
    function_name: str
    tool_call_id: str
    function_args_json: str


def get_first_usable_chunk(chat_completion_stream):
    chunk = next(chat_completion_stream)
    while not (has_function_call_id(chunk) or has_msg_content(chunk)):
        chunk = next(chat_completion_stream)
    return chunk


def has_msg_content(chunk):
    return chunk.choices[0].delta.content is not None


def has_function_call_id(chunk):
    return (
        chunk.choices[0].delta.tool_calls is not None
        and chunk.choices[0].delta.tool_calls[0].id is not None
    )


def extract_fns_from_chat(chat_completion_stream, first_chunk):
    function_name = first_chunk.choices[0].delta.tool_calls[0].function.name
    tool_call_id = first_chunk.choices[0].delta.tool_calls[0].id
    function_args_json = ""
    function_calls = []

    for chunk in chat_completion_stream:
        finish_reason = chunk.choices[0].finish_reason
        if finish_reason is not None:
            break
        elif has_function_call_id(chunk):
            function_calls.append(
                FunctionCall(
                    function_name=function_name,
                    tool_call_id=tool_call_id,
                    function_args_json=function_args_json,
                )
            )

            function_name = chunk.choices[0].delta.tool_calls[0].function.name
            tool_call_id = chunk.choices[0].delta.tool_calls[0].id
            function_args_json = ""
        else:
            function_args_json += (
                chunk.choices[0].delta.tool_calls[0].function.arguments
            )

    function_calls.append(
        FunctionCall(
            function_name=function_name,
            tool_call_id=tool_call_id,
            function_args_json=function_args_json,
        )
    )
    return function_calls

def get_user_input(use_audio_input, openai_client):
    if use_audio_input:
        audio_input = get_audio_input()
        text_input = get_text_from_speech(audio_input, openai_client)
        print(f"You: {text_input}")
    else:
        text_input = input("You: ")

    return text_input

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
    args = parser.parse_args()

    openai_client = OpenAI()
    elevenlabs_client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    create_db_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "hi, welcome to Heaven Coffee"},
    ]
    print("Assistant: hi, welcome to Heaven Coffee")
    need_user_input = True
    current_node_schema = take_order_node_schema
    current_edge_schemas = FROM_NODE_ID_TO_EDGE_SCHEMA[current_node_schema.id]
    new_node_input = None

    while True:
        if not current_node_schema.is_initialized:
            print(f"CURRENT_NODE_SCHEMA: {current_node_schema.id}")
            current_node_schema.run(new_node_input)
            print(current_node_schema.prompt)
            messages.append({"role": "system", "content": current_node_schema.prompt})

        if need_user_input:
            # Read user input from stdin
            text_input = get_user_input(args.audio_input, openai_client)
            # If user types 'exit', break the loop and end the program
            if text_input.lower() == "exit":
                print("Exiting chatbot. Goodbye!")
                break

            messages.append({"role": "user", "content": text_input})
        chat_completion = openai_client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=current_node_schema.tool_fns,
            stream=True,
        )

        first_chunk = get_first_usable_chunk(chat_completion)
        is_tool_call = has_function_call_id(first_chunk)

        if not is_tool_call:
            print("Assistant: ", end="")
            chat_stream_iterator = ChatCompletionIterator(chat_completion)
            if args.audio_output:
                get_speech_from_text(chat_stream_iterator, elevenlabs_client)
                print(chat_stream_iterator.full_msg)
            else:
                try:
                    while True:
                        print(next(chat_stream_iterator), end="")
                except StopIteration:
                    pass
                print()
            messages.append(
                {"role": "assistant", "content": chat_stream_iterator.full_msg}
            )
            need_user_input = True
        elif is_tool_call:
            function_calls = extract_fns_from_chat(chat_completion, first_chunk)

            for function_call in function_calls:
                function_args = json.loads(function_call.function_args_json)
                print(
                    f"[CALLING] {function_call.function_name} with args {function_args}"
                )
                if function_call.function_name.startswith("get_state"):
                    fn_output = getattr(
                        current_node_schema, function_call.function_name
                    )(**function_args)
                elif function_call.function_name.startswith("update_state"):
                    fn_output = current_node_schema.update_state(
                        **function_args
                    )
                    state_condition_results = [
                        edge_schema.state_condition_fn(current_node_schema.state)
                        for edge_schema in current_edge_schemas
                    ]
                    if any(
                        [
                            edge_schema.state_condition_fn(current_node_schema.state)
                            for edge_schema in current_edge_schemas
                        ]
                    ):
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

                function_call_result_msg = {
                    "role": "tool",
                    "content": json.dumps(fn_output, cls=CustomJSONEncoder),
                    "tool_call_id": function_call.tool_call_id,
                }
                print(
                    f"[CALLING DONE] {function_call.function_name} with output {fn_output}"
                )

                tool_call_message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": function_call.tool_call_id,
                            "type": "function",
                            "function": {
                                "arguments": function_call.function_args_json,
                                "name": function_call.function_name,
                            },
                        }
                    ],
                }

                messages.append(tool_call_message)
                messages.append(function_call_result_msg)
                if not function_call.function_name.startswith(
                    ("get_state", "update_state")
                ):
                    messages.append(
                        get_system_return_type_prompt(function_call.function_name)
                    )

                need_user_input = False