import queue
import threading
import time
import wave

import numpy as np
import pyaudio

from gui import remove_previous_line
from logger import logger

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
audio_queue = queue.Queue()
stop_recording_event = (
    threading.Event()
)  # Event to signal when to stop the recording thread


def is_silent(data):
    # Convert raw audio data to NumPy array to analyze sound levels
    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    # Compute the volume (RMS)
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return rms < SILENCE_THRESHOLD


def record_audio(stream):
    print("You: Recording...")
    while not stop_recording_event.is_set():
        data = stream.read(CHUNK)
        audio_queue.put(data)


def process_audio():
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


def get_audio_input():
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


def save_audio_to_wav(audio_data, file_path):
    """Save raw audio data to a .wav file."""
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
