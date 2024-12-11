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
    PatientFriendlySection,
    ClinicianFriendlySection,
    ReferencedItem,
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

FFMPEG_PATH = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\GENERAL UTILITIES\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
if not os.path.exists(FFMPEG_PATH):
    raise FileNotFoundError(f"ffmpeg not found at {FFMPEG_PATH}")

original_load_audio = whisper.audio.load_audio


def patched_load_audio(file: str, sr: int = whisper.audio.SAMPLE_RATE):
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
        proc = subprocess.run(cmd, capture_output=True, check=True)
        return (
            np.frombuffer(proc.stdout, np.int16).flatten().astype(np.float32) / 32768.0
        )
    return original_load_audio(file, sr)


whisper.audio.load_audio = patched_load_audio

DEFAULT_MODEL = "gpt-4o-mini"
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    return embedding_model(waveform[None])


def transcribe_and_diarize_audio(
    audio_path: str, output_path: str, num_speakers: int = 2
):
    base, ext = os.path.splitext(audio_path)
    if ext.lower() != ".wav":
        wav_path = base + ".wav"
        subprocess.run(
            [FFMPEG_PATH, "-i", audio_path, "-ac", "1", wav_path, "-y"], check=True
        )
        audio_path = wav_path

    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    segments = result["segments"]

    with contextlib.closing(wave.open(audio_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, audio_path, duration)
    embeddings = np.nan_to_num(embeddings)

    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_

    for i in range(len(segments)):
        segments[i]["speaker"] = "Speaker " + str(labels[i] + 1)

    def format_timestamp(seconds):
        return str(datetime.timedelta(seconds=int(seconds)))

    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
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

    # Prompt to extract entities as before
    prompt = f"""
You are a careful medical data extractor. Given a clinical conversation (turn by turn), identify:
- conditions
- medications
- instructions
- follow_up actions

For each entity:
- type (condition/medication/instruction/follow_up)
- name (if known)
- dosage (if medication)
- text (exact snippet)
- first_mention_turn (int)
- patient_explanation (simple explanation for patient)
- needs_clarification (bool)
- related_turns: all turn indices where this entity appears or is relevant

Also identify ambiguous terms if any.

Return EXACT JSON per schema:
{schema}

--- CONVERSATION TURNS ---
{turns_str}
---
"""
    return await handler.process(
        prompts=prompt, model=model_name, response_type=LLMTermExtractionResponse
    )


class EnhancedOutput(BaseModel):
    patient_friendly: PatientFriendlySection
    clinician_friendly: ClinicianFriendlySection


async def run_llm_clinician_insights(
    conversation_id: str, turns: list, entities: list, model_name: str
):
    handler = UnifiedLLMHandler()
    conversation_snippet = ""
    for i, turn in enumerate(turns):
        conversation_snippet += f"Turn {i}: {turn}\n"

    # Prepare a concise list of entities with turns
    entities_str = ""
    for e in entities:
        entities_str += f"- {e.type.upper()} Name:{e.name or 'N/A'}, Text:\"{e.text or ''}\", Turns:{e.related_turns}, PatientExplanation:\"{e.patient_explanation or ''}\"\n"

    schema = EnhancedOutput.model_json_schema()

    # New prompt demanding very specific structure referencing turns
    prompt = f"""
You are a medical assistant creating structured insights from a conversation.

Return EXACT JSON per schema:
{schema}

Description of required fields:
- patient_friendly.main_concerns: List of key patient concerns. Each item: title (short), description (patient-friendly), turns (list of turns referencing the transcript).
- patient_friendly.next_steps: List of recommended next steps for the patient, simple language. Each item includes turns where this advice was discussed.
- patient_friendly.medication_notes: For each medication mention, clarify usage. Each item references turns.
- patient_friendly.lifestyle_suggestions: Suggest lifestyle changes mentioned. Each references turns.

- clinician_friendly.assessment: Clinician-level assessment items. Each item: title, description (clinical detail), and turns.
- clinician_friendly.recommendations: Clinical recommendations (e.g., further tests). Each item references turns.
- clinician_friendly.ambiguities: Terms or instructions that need clarification. Each item references turns.
- clinician_friendly.differential_diagnoses: Potential diagnoses. Each references turns.
- clinician_friendly.diagnostic_steps: Tests or steps to confirm diagnosis. Each references turns.

Use the conversation and entities below. Ensure every item references the exact turns from which the info was derived.

--- CONVERSATION ---
{conversation_snippet}

--- ENTITIES ---
{entities_str}
---

Always follow schema strictly. If no information is available for a field, return an empty list for it.
"""

    insight_result = await handler.process(
        prompts=prompt, model=model_name, response_type=EnhancedOutput
    )
    if insight_result.success and insight_result.data:
        return (
            insight_result.data.patient_friendly,
            insight_result.data.clinician_friendly,
        )
    else:
        return PatientFriendlySection(), ClinicianFriendlySection()


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
                if e.first_mention_turn not in e.related_turns:
                    e.related_turns.append(e.first_mention_turn)

        summary = generate_executive_summary(entities)
        confusions = identify_confusions(turns)
        patient_friendly, clinician_friendly = await run_llm_clinician_insights(
            conversation_id, turns, entities, model_name
        )

        final_result = ConversationAnalysis(
            conversation_id=conversation_id,
            summary=summary,
            entities=entities,
            highlighted_confusions=confusions,
            patient_friendly=patient_friendly,
            clinician_friendly=clinician_friendly,
        ).model_dump(mode="json")

        return final_result

    except Exception as e:
        print(traceback.format_exc())
        return {
            "conversation_id": conversation_id,
            "error": f"Processing failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }
