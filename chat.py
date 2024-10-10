import argparse
import json
import os
import tempfile
from collections.abc import Iterator
from distutils.util import strtobool

from dotenv import load_dotenv  # Add this import
from elevenlabs import ElevenLabs, Voice, VoiceSettings, stream
from openai import OpenAI

from audio import get_audio_input, save_audio_to_wav
from chain import FROM_NODE_ID_TO_EDGE_SCHEMA, take_order_node_schema
from db_functions import (
    FN_NAME_TO_FN,
    OPENAI_TOOLS_RETUN_DESCRIPTION,
    create_client,
    obj_to_dict,
)

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
    args = parser.parse_args()

    openai_client = OpenAI()
    elevenlabs_client = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    create_client()

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
            if args.audio_input:
                audio_input = get_audio_input()
                text_input = get_text_from_speech(audio_input, openai_client)
                print(f"You: {text_input}")
            else:
                text_input = input("You: ")
            # If user types 'exit', break the loop and end the program
            if text_input.lower() == "exit":
                print("Exiting chatbot. Goodbye!")
                break

            messages.append({"role": "user", "content": text_input})

        chat_completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=current_node_schema.tool_fns,
            stream=True,
        )
        first_chunk = next(chat_completion)
        is_tool_call = first_chunk.choices[0].delta.tool_calls

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
            function_name = first_chunk.choices[0].delta.tool_calls[0].function.name
            tool_call_id = first_chunk.choices[0].delta.tool_calls[0].id
            function_args_json = ""
            for chunk in chat_completion:
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason is not None:
                    break
                function_args_json += (
                    chunk.choices[0].delta.tool_calls[0].function.arguments
                )

            fuction_args = json.loads(function_args_json)
            print(f"[CALLING] {function_name} with args {fuction_args}")
            if function_name in ["get_state", "update_state"]:
                fn_output = getattr(current_node_schema, function_name)(**fuction_args)
                if function_name == "update_state":
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
                fn = FN_NAME_TO_FN[function_name]
                fn_output = fn(**fuction_args)

            fn_output = obj_to_dict(fn_output)
            function_call_result_msg = {
                "role": "tool",
                "content": json.dumps(fn_output),
                "tool_call_id": tool_call_id,
            }
            print(f"[CALLING DONE] {function_name} with output {fn_output}")

            tool_call_message = {
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

            messages.append(tool_call_message)
            messages.append(function_call_result_msg)
            if function_name not in ["get_state", "update_state"]:
                messages.append(get_system_return_type_prompt(function_name))

            need_user_input = False
