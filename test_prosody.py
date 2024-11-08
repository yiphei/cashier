import asyncio
import base64
import io
import json
import queue
import threading
import time
import wave

import numpy as np
import pyaudio
import websockets

from logger import logger

# Settings for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2

# Thread-safe queues for communication
audio_queue = queue.Queue()
stop_recording_event = threading.Event()


def is_silent(data):
    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return rms < SILENCE_THRESHOLD


def create_wav_file(audio_data):
    """Create a WAV file in memory from raw audio data"""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wav_file.setframerate(RATE)
        wav_file.writeframes(audio_data)
    return wav_buffer.getvalue()


def record_audio(stream):
    """Record audio in a separate thread and put data in the queue"""
    print("You: Recording...")
    silence_start = None

    # Buffer to accumulate chunks for WAV creation
    audio_buffer = b""

    while not stop_recording_event.is_set():
        data = stream.read(CHUNK)
        if not is_silent(data):
            # Accumulate the audio data
            audio_buffer += data

            # Create and send WAV file every ~500ms (or adjust as needed)
            if len(audio_buffer) >= RATE:  # 16000 bytes = 1 second of audio
                wav_data = create_wav_file(audio_buffer)
                encoded_data = base64.b64encode(wav_data).decode("utf-8")
                audio_queue.put(encoded_data)
                audio_buffer = b""  # Reset buffer

            silence_start = None
        else:
            silence_start = silence_start or time.time()
            if time.time() - silence_start > SILENCE_DURATION:
                # Send any remaining audio data before stopping
                if audio_buffer:
                    wav_data = create_wav_file(audio_buffer)
                    encoded_data = base64.b64encode(wav_data).decode("utf-8")
                    audio_queue.put(encoded_data)
                stop_recording_event.set()


async def process_audio_queue(websocket):
    """Process audio data from the queue and send it over websocket"""
    while not stop_recording_event.is_set() or not audio_queue.empty():
        try:
            # Get data from queue with a timeout to allow checking stop_recording_event
            encoded_data = audio_queue.get(timeout=0.1)
            message = {"models": {"prosody": {}}, "data": encoded_data}
            await websocket.send(json.dumps(message))
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            break


async def stream_audio_to_hume():
    uri = "wss://api.hume.ai/v0/stream/models"
    headers = {"X-Hume-Api-Key": "b9G8cUfkrO8WMOrAanOILg4AGDKGAZcjUSEALNOhClRdsvSD"}

    async with websockets.connect(uri, extra_headers=headers) as websocket:
        print("Connected to Hume AI WebSocket")

        # Initialize audio
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        # Start recording thread
        record_thread = threading.Thread(target=record_audio, args=(stream,))
        record_thread.start()

        # Create tasks for sending and receiving
        send_task = asyncio.create_task(process_audio_queue(websocket))

        try:
            # Process incoming messages
            async for response in websocket:
                result = json.loads(response)
                prosody = result["prosody"]
                if "predictions" in prosody:
                    emotions = prosody["predictions"][0]["emotions"]
                    sorted_emotions = sorted(
                        emotions, key=lambda item: float(item["score"]), reverse=True
                    )
                    logger.debug(f"Received response: {sorted_emotions[:3]}")
                else:
                    logger.debug(f"Received response: {result}")
        except Exception as e:
            logger.error(f"Error in websocket communication: {e}")
        finally:
            # Cleanup
            stop_recording_event.set()
            record_thread.join()
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass

            stream.stop_stream()
            stream.close()
            audio.terminate()
            audio_queue.queue.clear()
            stop_recording_event.clear()


if __name__ == "__main__":
    asyncio.run(stream_audio_to_hume())
