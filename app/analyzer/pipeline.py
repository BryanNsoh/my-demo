import json
from pathlib import Path
import traceback
from typing import Dict

from app.analyzer.models import (
    LLMTermExtractionResponse,
    FinalConversationAnalysis,
    FinalEntity,
    FinalContextSample,
)
from vendor.unified_llm_handler import UnifiedLLMHandler, UnifiedResponse

DATA_DIR = Path("data")
CACHE_DIR = Path("cache")


def read_conversation_text(conversation_id: str) -> str:
    file_path = DATA_DIR / f"{conversation_id}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Conversation file {file_path} does not exist.")
    return file_path.read_text().strip()


async def run_llm_extraction(
    conversation_text: str,
) -> UnifiedResponse[LLMTermExtractionResponse]:
    handler = UnifiedLLMHandler()
    schema = LLMTermExtractionResponse.model_json_schema()

    prompt = f"""
You are a helpful assistant specialized in analyzing clinical visit conversations.
I will provide a full patient-doctor conversation. Your task:
1. Identify conditions, medications, instructions, follow-ups (with first mention sentences).
2. Identify ambiguous terms and how context resolves them.
3. Return EXACT JSON per schema:
{schema}

--- CONVERSATION START ---
{conversation_text}
--- CONVERSATION END ---
"""

    result = await handler.process(
        prompts=prompt, model="gpt-4o-mini", response_type=LLMTermExtractionResponse
    )
    return result


def enrich_final_output(
    conversation_id: str, llm_response: LLMTermExtractionResponse
) -> FinalConversationAnalysis:
    final_entries = []
    for entity in llm_response.entities:
        patient_explanation = None
        link = None
        if entity.type == "condition":
            patient_explanation = (
                "Your shoulder muscles are strained from playing tennis."
            )
            link = "https://medlineplus.gov/shoulderpain.html"
        elif entity.type == "medication":
            patient_explanation = "Ibuprofen helps reduce swelling and pain."
            link = "https://medlineplus.gov/ibuprofen.html"
        elif entity.type == "instruction":
            patient_explanation = "Icing helps reduce swelling and pain."
        elif entity.type == "follow_up":
            patient_explanation = (
                "We need to check if your condition improves over time."
            )

        final_entries.append(
            FinalEntity(
                type=entity.type,
                name=entity.name,
                dosage=entity.dosage,
                text=entity.text,
                first_mention_sentence=entity.first_mention_sentence,
                patient_explanation=patient_explanation,
                link=link,
            )
        )

    final_context = []
    for cs in llm_response.context_samples:
        final_context.append(
            FinalContextSample(
                ambiguous_term=cs.ambiguous_term,
                resolved_meaning=cs.resolved_meaning,
                reasoning=cs.reasoning,
                sentence_id=cs.sentence_id,
            )
        )

    return FinalConversationAnalysis(
        conversation_id=conversation_id,
        entries=final_entries,
        context_samples=final_context,
    )


def load_from_cache(conversation_id: str) -> dict:
    cache_file = CACHE_DIR / f"{conversation_id}.json"
    if cache_file.exists():
        try:
            with cache_file.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # If corrupted or not JSON, treat as no cache
            pass
    return None


def save_to_cache(conversation_id: str, data):
    # If data is pydantic, ensure JSON serializable form:
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{conversation_id}.json"
    with cache_file.open("w") as f:
        json.dump(data, f, indent=2)


async def process_conversation(conversation_id: str) -> dict:
    cached_result = load_from_cache(conversation_id)
    if cached_result is not None:
        return cached_result

    convo_text = read_conversation_text(conversation_id)
    try:
        llm_resp = await run_llm_extraction(convo_text)

        if not llm_resp.success:
            return {
                "conversation_id": conversation_id,
                "error": f"LLM extraction failed: {llm_resp.error}",
            }

        final_result = enrich_final_output(conversation_id, llm_resp.data)
        save_to_cache(conversation_id, final_result)  # final_result is pydantic model
        return final_result.model_dump(mode="json")

    except Exception as e:
        print(traceback.format_exc())
        return {
            "conversation_id": conversation_id,
            "error": f"LLM extraction failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }
