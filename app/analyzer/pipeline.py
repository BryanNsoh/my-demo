import json
import traceback
import hashlib
from pathlib import Path
from typing import Optional
from app.analyzer.models import (
    LLMTermExtractionResponse,
    ConversationAnalysis,
    ExecutiveSummary,
    ExtractedEntity,
)
from vendor.unified_llm_handler import UnifiedLLMHandler, UnifiedResponse

import os
import datetime
import subprocess
import torch
import wave
import contextlib
import numpy as np
import shutil
from pydantic import BaseModel, Field

# Monkey-patch speechbrain symlink issue
import speechbrain.utils.fetching


def always_copy(src, dst, strategy):
    shutil.copyfile(src, dst)
    return dst


speechbrain.utils.fetching.link_with_strategy = always_copy

from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import whisper
import whisper.audio

# Configure exact ffmpeg path
FFMPEG_PATH = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\GENERAL UTILITIES\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

if not os.path.exists(FFMPEG_PATH):
    raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_PATH}")

# Monkey-patch whisper's audio loading to use absolute ffmpeg path
original_load_audio = whisper.audio.load_audio


def patched_load_audio(file: str, sr: int = whisper.audio.SAMPLE_RATE):
    """Patched version of whisper's load_audio that uses absolute ffmpeg path"""
    if isinstance(file, str):
        cmd = [
            FFMPEG_PATH,
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, check=True)
            return (
                np.frombuffer(proc.stdout, np.int16).flatten().astype(np.float32)
                / 32768.0
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}")

    return original_load_audio(file, sr)


# Replace whisper's load_audio with our patched version
whisper.audio.load_audio = patched_load_audio

# Global default model name
DEFAULT_MODEL = "gpt-4o-mini"

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize models and audio processors
audio = Audio()

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cpu")
)


def read_conversation_text(conversation_id: str) -> str:
    file_path_txt = DATA_DIR / f"{conversation_id}.txt"
    if file_path_txt.exists():
        return file_path_txt.read_text(encoding="utf-8").strip()
    return ""


def split_into_turns(conversation_text: str) -> list:
    lines = [l.strip() for l in conversation_text.split("\n") if l.strip() != ""]
    return lines


def segment_embedding(segment, path, duration):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    # No indexing needed, matches the working example
    return embedding_model(waveform[None])


def transcribe_and_diarize_audio(
    audio_path: str, output_path: str, num_speakers: int = 2
):
    # Convert to mono WAV if needed
    base, ext = os.path.splitext(audio_path)
    if ext.lower() != ".wav":
        wav_path = base + ".wav"
        # Force mono conversion during ffmpeg
        subprocess.run(
            [FFMPEG_PATH, "-i", audio_path, "-ac", "1", wav_path, "-y"], check=True
        )
        audio_path = wav_path

    # Load and transcribe with whisper
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    segments = result["segments"]

    # Get audio duration
    with contextlib.closing(wave.open(audio_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # Generate embeddings for each segment
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, audio_path, duration)
    embeddings = np.nan_to_num(embeddings)

    # Cluster the embeddings
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_

    # Add speaker labels to segments
    for i in range(len(segments)):
        segments[i]["speaker"] = "Speaker " + str(labels[i] + 1)

    # Write output with speaker labels and timestamps
    def format_timestamp(seconds):
        return str(datetime.timedelta(seconds=int(seconds)))

    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            # Add speaker marker when speaker changes or at start
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write(
                    f"\n{segment['speaker']} [{format_timestamp(segment['start'])}]:\n"
                )
            f.write(segment["text"].strip() + " ")


async def run_llm_extraction(
    conversation_id: str, turns: list, model_name: str
) -> UnifiedResponse[LLMTermExtractionResponse]:
    handler = UnifiedLLMHandler()
    schema = LLMTermExtractionResponse.model_json_schema()
    turns_str = ""
    for i, turn in enumerate(turns):
        turns_str += f"Turn {i}: {turn}\n"

    prompt = f"""
You are a helpful medical assistant analyzing a clinical visit conversation. I will provide the conversation broken down by turns.

Your tasks:
1. Identify conditions, medications, instructions, follow_up actions.
2. For each entity, provide type, name if known, dosage if medication, text, first_mention_turn, patient_explanation, needs_clarification.
3. Identify ambiguous terms if any.

Return EXACT JSON per schema:
{schema}

--- CONVERSATION TURNS ---
{turns_str}
--- END OF CONVERSATION ---
"""
    result = await handler.process(
        prompts=prompt, model=model_name, response_type=LLMTermExtractionResponse
    )
    return result


async def run_llm_clinician_insights(
    conversation_id: str, turns: list, entities: list, model_name: str
) -> list:
    handler = UnifiedLLMHandler()
    conversation_snippet = ""
    for i, turn in enumerate(turns):
        conversation_snippet += f"Turn {i}: {turn}\n"
    entity_summary = "Extracted Entities:\n"
    for e in entities:
        entity_summary += (
            f"- {e.type}: {e.text}, explanation: {e.patient_explanation}\n"
        )

    class ClinicianInsightsResponse(BaseModel):
        insights: list[str] = Field(default_factory=list)

    schema = ClinicianInsightsResponse.model_json_schema()
    prompt = f"""
Return EXACT JSON per schema:
{schema}

You are assisting a clinician. Given the conversation and extracted entities, produce 3-5 concise bullet-point insights that a clinician would find helpful:
- Possible conditions/differentials
- Medication adherence issues
- Key lifestyle factors
- Next diagnostic steps

--- CONVERSATION ---
{conversation_snippet}
--- ENTITIES ---
{entity_summary}
"""

    insight_result = await handler.process(
        prompts=prompt, model=model_name, response_type=ClinicianInsightsResponse
    )
    if insight_result.success and insight_result.data:
        return insight_result.data.insights
    else:
        return []


def generate_executive_summary(entities: list) -> ExecutiveSummary:
    main_concerns = [e.text for e in entities if e.type == "condition"]
    critical_steps = [e.text for e in entities if e.type == "follow_up"]
    return ExecutiveSummary(
        main_concerns=main_concerns[:3], critical_next_steps=critical_steps[:3]
    )


def identify_confusions(turns: list) -> list:
    confusion_keywords = ["not sure", "don't know", "confused", "unsure", "which one"]
    confusions = []
    for i, turn in enumerate(turns):
        if any(kw in turn.lower() for kw in confusion_keywords):
            confusions.append(turn)
    return confusions


async def process_conversation(conversation_id: str, model_name: str) -> dict:
    txt_path = DATA_DIR / f"{conversation_id}.txt"
    audio_path_mp3 = DATA_DIR / f"{conversation_id}.mp3"
    audio_path_wav = DATA_DIR / f"{conversation_id}.wav"

    try:
        if audio_path_mp3.exists() or audio_path_wav.exists():
            audio_file_path = (
                str(audio_path_mp3) if audio_path_mp3.exists() else str(audio_path_wav)
            )
            output_text_path = str(txt_path)
            transcribe_and_diarize_audio(audio_file_path, output_text_path)
        elif not txt_path.exists():
            return {
                "conversation_id": conversation_id,
                "error": f"No input data for {conversation_id}",
            }

        convo_text = read_conversation_text(conversation_id)
        turns = split_into_turns(convo_text)

        llm_resp = await run_llm_extraction(conversation_id, turns, model_name)
        if not llm_resp.success:
            return {
                "conversation_id": conversation_id,
                "error": f"LLM extraction failed: {llm_resp.error}",
            }

        entities = llm_resp.data.entities
        for e in entities:
            if e.first_mention_turn is not None:
                e.related_turns.append(e.first_mention_turn)

        summary = generate_executive_summary(entities)
        confusions = identify_confusions(turns)
        insights = await run_llm_clinician_insights(
            conversation_id, turns, entities, model_name
        )

        final_result = ConversationAnalysis(
            conversation_id=conversation_id,
            summary=summary,
            entities=entities,
            highlighted_confusions=confusions,
            clinician_insights=insights,
        ).model_dump(mode="json")

        return final_result

    except Exception as e:
        print(traceback.format_exc())
        return {
            "conversation_id": conversation_id,
            "error": f"Processing failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }
