import json
import os
import tempfile
from collections.abc import Iterator

from dotenv import load_dotenv  # Add this import
from elevenlabs import ElevenLabs, Voice, VoiceSettings, stream
from openai import OpenAI

from audio import get_audio_input, save_audio_to_wav
from db_functions import (
    OPENAI_TOOLS,
    OPENAI_TOOLS_RETUN_DESCRIPTION,
    create_client,
    get_menu_item_from_name,
    get_menu_items_options,
)

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = (
    "You are a cashier working for the coffee shop Heaven Coffee, and you are physically embedded in it, "
    "meaning you will interact with real in-person customers. There is a microphone that transcribes customer's speech to text, "
    "and a speaker that outputs your text to speech. Because your responses will be converted to speech, "
    "you must respond in a conversational way: natural and easy to understand when converted to speech. So do not use "
    "any text formatting like hashtags, bold, italic, bullet points, etc. \n\n"
    "Customers come to you to place orders. "
    "Your job is to take their orders, answer reasonable questions about the shop & menu only, and assist "
    "them with any issues they may have about their orders. You are not responsible for anything else, "
    "so you must refuse to engage in anything unrelated. However, do not refuse too explicitly, abruptly, or rudely."
    "Instead, be sensitive and engage in small talk when necessary, as long as it quickly leads to the main business."
    "If they dont work, then you can progressively refuse more firmly."
    "Overall, be professional, polite, and friendly."
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
            self.full_msg += msg  # Append the message to full_msg
            return msg  # Return the message
        except StopIteration:
            raise StopIteration  # Signal end of iteration


def get_system_return_type_prompt(fn_name):
    description = OPENAI_TOOLS_RETUN_DESCRIPTION[fn_name]
    msg = {
        "role": "system",
        "content": f"This is the JSON Schema of {fn_name}'s return type: {json.dumps(description)}",
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

    while True:
        if need_user_input:
            # Read user input from stdin

            audio_input = get_audio_input()
            text_input = get_text_from_speech(audio_input, openai_client)
            print(f"You: {text_input}")
            # If user types 'exit', break the loop and end the program
            if text_input.lower() == "exit":
                print("Exiting chatbot. Goodbye!")
                break

            messages.append({"role": "user", "content": text_input})

        chat_completion = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=OPENAI_TOOLS, stream=True
        )
        first_chunk = next(chat_completion)
        is_tool_call = first_chunk.choices[0].delta.tool_calls

        if not is_tool_call:
            print("Assistant: ", end="")
            chat_stream_iterator = ChatCompletionIterator(chat_completion)
            get_speech_from_text(chat_stream_iterator, elevenlabs_client)
            print(chat_stream_iterator.full_msg)
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
            if function_name == "get_menu_item_from_name":
                menu_item = get_menu_item_from_name(**fuction_args)
                content = menu_item.model_dump()
            else:
                mapping = get_menu_items_options(**fuction_args)
                content = {
                    k: [sub_v.model_dump() for sub_v in v] for k, v in mapping.items()
                }

            function_call_result_msg = {
                "role": "tool",
                "content": json.dumps(content),
                "tool_call_id": tool_call_id,
            }
            print(f"[CALLING DONE] {function_name} with output {content}")

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
            messages.append(get_system_return_type_prompt(function_name))

            need_user_input = False
