import json
import os
import tempfile

from dotenv import load_dotenv  # Add this import
from elevenlabs import ElevenLabs, Voice, VoiceSettings, play
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
    "and a speaker that outputs your text to speech. Customers come to you to place orders. "
    "Your job is to take their orders, answer reasonable questions about the shop & menu only, and assist "
    "them with any issues they may have about their orders. You are not responsible for anything else, "
    "so you must refuse to engage in anything unrelated."
)


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


def get_speech_from_text(text, client):
    audio = client.generate(
        voice=Voice(
            voice_id="cgSgspJ2msm6clMCkdW9",
            settings=VoiceSettings(
                stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
            ),
        ),
        text=text,
        model="eleven_multilingual_v2",
    )
    play(audio)


if __name__ == "__main__":
    client = OpenAI()
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

            audio_data = get_audio_input()
            user_input = get_text_from_speech(audio_data, client)
            print(f"You: {user_input}")
            # If user types 'exit', break the loop and end the program
            if user_input.lower() == "exit":
                print("Exiting chatbot. Goodbye!")
                break

            messages.append({"role": "user", "content": user_input})

        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=OPENAI_TOOLS
        )

        response = chat_completion.choices[0]
        finish_reason = response.finish_reason
        if finish_reason == "stop":
            response_msg = response.message.content
            print("Assistant: ", response_msg)
            get_speech_from_text(response_msg, elevenlabs_client)
            messages.append({"role": "assistant", "content": response_msg})

            need_user_input = True
        elif finish_reason == "tool_calls":
            tool_call_message = response.message
            function_name = response.message.tool_calls[0].function.name
            fuction_args = json.loads(response.message.tool_calls[0].function.arguments)
            tool_call_id = response.message.tool_calls[0].id
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

            messages.append(tool_call_message)
            messages.append(function_call_result_msg)
            messages.append(get_system_return_type_prompt(function_name))

            need_user_input = False
