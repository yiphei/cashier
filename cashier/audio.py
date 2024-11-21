import queue
import tempfile
import threading
import time
import wave
from typing import Any, Iterator, Union

import numpy as np
import pyaudio
from elevenlabs import Voice, VoiceSettings, stream

from cashier.gui import remove_previous_line
from cashier.logger import logger
from cashier.model.model_other import ModelClient
from cashier.model.model_util import ModelProvider

# Settings for audio recording
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Mono
RATE = 16000  # 16kHz sample rate, adjust to your model's needs
CHUNK = 1024  # Buffer size
SILENCE_THRESHOLD = (
    500  # Threshold for considering silence (adjust based on environment)
)
SILENCE_DURATION = 2  # Time in seconds to consider the recording stopped after silence

# Thread-safe queue to hold recorded frames
audio_queue: queue.Queue[bytes] = queue.Queue()
stop_recording_event = (
    threading.Event()
)  # Event to signal when to stop the recording thread


def is_silent(data: bytes) -> bool:
    # Convert raw audio data to NumPy array to analyze sound levels
    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    # Compute the volume (RMS)
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return rms < SILENCE_THRESHOLD


def record_audio(stream: pyaudio.Stream) -> None:
    print("You: Recording...")
    while not stop_recording_event.is_set():
        data = stream.read(CHUNK)
        audio_queue.put(data)


def process_audio() -> bytes:
    frames = []
    has_spoken = False
    silence_start = None

    while True:
        if not audio_queue.empty():
            data = audio_queue.get()
            frames.append(data)

            if is_silent(data):
                # If silence starts, note the time
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION and has_spoken:
                    # If silence has lasted long enough, stop recording
                    remove_previous_line()
                    logger.debug("[AUDIO] Silence detected, stopping recording.")
                    stop_recording_event.set()  # Signal the recording thread to stop
                    break
            else:
                # If non-silent data, reset silence timer
                silence_start = None
                has_spoken = True

    logger.debug("[AUDIO] Finished recording.")
    return b"".join(frames)  # Return raw audio data


def get_audio_input() -> bytes:
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    # Start recording in a separate thread
    record_thread = threading.Thread(target=record_audio, args=(stream,))
    record_thread.start()
    # Process audio in the main thread
    audio_data = process_audio()

    # Wait for the recording thread to finish
    record_thread.join()

    stream.stop_stream()
    stream.close()
    audio.terminate()
    audio_queue.queue.clear()
    stop_recording_event.clear()
    return audio_data


def save_audio_to_wav(audio_data: bytes, file_path: str) -> None:
    """Save raw audio data to a .wav file."""
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio_data)


def get_text_from_speech(audio_data: bytes, oai_client: Any) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav_file:
        # Save the audio data as a WAV file
        save_audio_to_wav(audio_data, temp_wav_file.name)

        # Use OpenAI's API to transcribe the saved WAV file
        transcription = ModelClient.get_client(ModelProvider.OPENAI).audio.transcriptions.create(
            model="whisper-1",
            language="en",
            file=open(temp_wav_file.name, "rb"),  # Open the saved WAV file for reading
        )
    return transcription.text


def get_speech_from_text(
    text_iterator: Union[str, Iterator[str]], elabs_client: Any
) -> None:
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
