from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from pydub import AudioSegment
from typing import Dict, Any
import base64
import json
import librosa
import logging
import numpy as np
import os
import torch
import uvicorn

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for reproducible demo results
# Note: TEN-framework/TEN_Turn_Detection must be fine tuned for production use
torch.manual_seed(999)

semantic_evaluator = None


class SemanticEvaluator:
    def __init__(self):
        self.model_name = "TEN-framework/TEN_Turn_Detection"
        self.cache_dir = "./models"

        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )

    def evaluate(self, text: str) -> float:
        """
        Args:
            text: The input sentence/utterance

        Returns:
            float: Probability that the speaker is finished speaking (0 - 1)
        """
        # Tokenize the input
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # get probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Check how many classes we have
            num_classes = probabilities.shape[-1]

            if num_classes == 2:
                prob_finished = probabilities[0][0].item()
                prob_continue = probabilities[0][1].item()

            else:
                raise ValueError(
                    f"Expected binary classification (2 classes), got {num_classes} classes"
                )

        return prob_finished


whisper_transcriber = None


class WhisperTranscriber:
    def __init__(self):
        logger.info("Whisper model is loading")
        self.cache_dir = "./models"
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v3", cache_dir=self.cache_dir
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3", cache_dir=self.cache_dir
        )

        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA acceleration")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        self.model.to(self.device)

        logger.info("Whisper model loaded successfully")

    def transcribe_audio(self, audio_data, sample_rate=8000):
        try:
            # Ensure float32 and normalized to [-1, 1]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Resample to 16kHz if needed (Whisper's expected sample rate)
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=16000
                )

            # Process audio with the processor
            input_features = self.processor(
                audio_data, sampling_rate=16000, return_tensors="pt"
            ).input_features

            # Move to device
            input_features = input_features.to(self.device)

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,  # Maximum sequence length
                    num_beams=1,  # Greedy decoding for speed
                    do_sample=False,
                )

            # Decode the prediction
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            return transcription.strip()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""


class MediaStreamHandler:
    def __init__(self):
        self.stream_sid = None
        self.audio_chunks = []

    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming Twilio Media Stream messages"""
        event = message.get("event")

        if event == "connected":
            logger.info("Media stream connected")

        elif event == "start":
            self.stream_sid = message.get("start", {}).get("streamSid")
            logger.info(f"Media stream started: {self.stream_sid}")

        elif event == "media":
            await self.handle_media(websocket, message)

        elif event == "stop":
            logger.info("Media stream stopped")

    async def handle_media(self, websocket: WebSocket, message: Dict[str, Any]):
        """Process incoming audio data"""
        try:
            payload = message.get("media", {}).get("payload", "")

            audio_bytes = base64.b64decode(payload)

            # Convert mu-law to linear PCM
            audio_segment = AudioSegment(
                data=audio_bytes, sample_width=1, frame_rate=8000, channels=1
            )

            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0  # Normalize to [-1, 1] range

            self.audio_chunks.extend(samples)

            # If we have enough audio (2 seconds), transcribe
            if len(self.audio_chunks) >= 16000:  # 2 seconds at 8kHz
                audio_array = np.array(self.audio_chunks)

                # Transcribe
                transcript = whisper_transcriber.transcribe_audio(audio_array)

                # Log it
                logger.info(f"TRANSCRIPT: {transcript}")

                # Clear chunks
                self.audio_chunks = []

        except Exception as e:
            logger.error(f"Error processing media: {e}")


app = FastAPI(title="Semantic Endpointing API")


@app.post("/incoming-call")
async def handle_incoming_call(request: Request):
    """Generate TwiML to connect to Media Stream WebSocket"""

    # Get hostname from environment variable
    HOSTNAME = os.getenv("HOSTNAME")

    # Generate TwiML response
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Hello! I'm connecting you to our semantic endpointing system.</Say>
    <Connect>
        <Stream url="wss://{HOSTNAME}/media-stream" />
    </Connect>
</Response>"""

    logger.info(f"Generated TwiML for media stream: {HOSTNAME}/media-stream")

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Twilio Media Stream"""
    media_handler = MediaStreamHandler()

    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        while True:
            # Receive message from Twilio
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle the message
            await media_handler.handle_message(websocket, message)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")


if __name__ == "__main__":
    semantic_evaluator = SemanticEvaluator()
    whisper_transcriber = WhisperTranscriber()

    # Get port from environment or default to 8080
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on port:{port}")
    logger.info(f"Hostname: {os.getenv('HOSTNAME', 'Not set')}")

    uvicorn.run(app, host=host, port=port)
