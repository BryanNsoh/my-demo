import json
from pathlib import Path
import traceback
from typing import Optional
from app.analyzer.models import (
    LLMTermExtractionResponse,
    FinalConversationAnalysis,
    ExtractedEntity,
    AmbiguousTermResolution,
)
from vendor.unified_llm_handler import UnifiedLLMHandler, UnifiedResponse

DATA_DIR = Path("data")
CACHE_DIR = Path("cache")


def read_conversation_text(conversation_id: str) -> str:
    file_path = DATA_DIR / f"{conversation_id}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Conversation file {file_path} does not exist.")
    return file_path.read_text().strip()


def split_into_turns(conversation_text: str) -> (str, list):
    # Split by lines, filter empty lines, each non-empty line is a turn
    lines = [l.strip() for l in conversation_text.split("\n") if l.strip() != ""]
    # Each line is now considered a turn
    # We'll present these turns to the LLM as a list with indices
    # Return the lines and we rely on LLM to reference them by index
    return lines


async def run_llm_extraction(
    conversation_id: str, turns: list
) -> UnifiedResponse[LLMTermExtractionResponse]:
    handler = UnifiedLLMHandler()
    schema = LLMTermExtractionResponse.model_json_schema()

    # We'll show turns with their indices. The LLM must refer to turn_id for each entity and ambiguous term.
    turns_str = ""
    for i, turn in enumerate(turns):
        # Example format: "Turn 0: Dr. Patel: Good morning..."
        turns_str += f"Turn {i}: {turn}\n"

    prompt = f"""
You are a helpful medical assistant analyzing a clinical visit conversation. I will provide the conversation broken down by turns, each turn has an index.
Your tasks:
1. Identify all:
   - conditions (health issues like heartburn, nausea)
   - medications (pills, drugs mentioned)
   - instructions (advice given like 'reduce salt', 'take medicine')
   - follow_up actions (like 'schedule appointment, do EKG, run tests')
2. For each entity, provide:
   - type: one of condition|medication|instruction|follow_up
   - name: if condition or medication has a known name (if unknown, best guess)
   - dosage if medication (if mentioned)
   - text if instruction/follow_up (the actual instruction text)
   - first_mention_turn (0-based turn index)
   - patient_explanation: a patient-friendly explanation of this entity. E.g., for a condition: what it is in simple terms; for a medication: what it does; for instruction/follow_up: why it's important.
   - link: a trusted health resource link (like a relevant MedlinePlus page) for conditions/medications if possible. For instructions/follow-ups, link can be omitted or set to null if none appropriate.
3. Identify ambiguous terms and show how context resolves them:
   - For each ambiguous term, provide turn_id where it first appears and explain reasoning.
4. Return EXACT JSON per schema below. No extra commentary.
{schema}

--- CONVERSATION TURNS ---
{turns_str}
--- END OF CONVERSATION ---
"""

    result = await handler.process(
        prompts=prompt, model="gpt-4o-mini", response_type=LLMTermExtractionResponse
    )
    return result


def load_from_cache(conversation_id: str) -> Optional[dict]:
    cache_file = CACHE_DIR / f"{conversation_id}.json"
    if cache_file.exists():
        try:
            with cache_file.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return None


def save_to_cache(conversation_id: str, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{conversation_id}.json"
    # data should be JSON-serializable (Pydantic model_dump or already dict)
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    with cache_file.open("w") as f:
        json.dump(data, f, indent=2)


async def process_conversation(conversation_id: str) -> dict:
    cached_result = load_from_cache(conversation_id)
    if cached_result is not None:
        return cached_result

    convo_text = read_conversation_text(conversation_id)
    turns = split_into_turns(convo_text)

    try:
        llm_resp = await run_llm_extraction(conversation_id, turns)

        if not llm_resp.success:
            return {
                "conversation_id": conversation_id,
                "error": f"LLM extraction failed: {llm_resp.error}",
            }

        # llm_resp.data is LLMTermExtractionResponse
        # We no longer do second pass. Just finalize into FinalConversationAnalysis
        final_result = FinalConversationAnalysis(
            conversation_id=conversation_id,
            entries=llm_resp.data.entities,
            context_samples=llm_resp.data.context_samples,
        )
        final_dict = final_result.model_dump(mode="json")
        save_to_cache(conversation_id, final_dict)
        return final_dict

    except Exception as e:
        print(traceback.format_exc())
        return {
            "conversation_id": conversation_id,
            "error": f"LLM extraction failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }
