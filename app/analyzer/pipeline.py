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

DATA_DIR = Path("data")
CACHE_DIR = Path("cache")


def read_conversation_text(conversation_id: str) -> str:
    file_path = DATA_DIR / f"{conversation_id}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Conversation file {file_path} does not exist.")
    return file_path.read_text().strip()


def split_into_turns(conversation_text: str) -> list:
    lines = [l.strip() for l in conversation_text.split("\n") if l.strip() != ""]
    return lines


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
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    with cache_file.open("w") as f:
        json.dump(data, f, indent=2)


def get_transcript_hash(conversation_text: str) -> str:
    return hashlib.sha256(conversation_text.encode("utf-8")).hexdigest()


async def run_llm_extraction(
    conversation_id: str, turns: list
) -> UnifiedResponse[LLMTermExtractionResponse]:
    handler = UnifiedLLMHandler()
    schema = LLMTermExtractionResponse.model_json_schema()

    turns_str = ""
    for i, turn in enumerate(turns):
        turns_str += f"Turn {i}: {turn}\n"

    prompt = f"""
You are a helpful medical assistant analyzing a clinical visit conversation. I will provide the conversation broken down by turns.

Your tasks:
1. Identify:
   - conditions (health issues)
   - medications
   - instructions (advice)
   - follow_up actions
2. For each entity, provide:
   - type
   - name if known
   - dosage if medication
   - text
   - first_mention_turn
   - patient_explanation (simple)
   - needs_clarification

3. Identify ambiguous terms if any.

Return EXACT JSON per schema:
{schema}

--- CONVERSATION TURNS ---
{turns_str}
--- END OF CONVERSATION ---
"""

    result = await handler.process(
        prompts=prompt, model="gpt-4o-mini", response_type=LLMTermExtractionResponse
    )
    return result


async def run_llm_clinician_insights(
    conversation_id: str, turns: list, entities: list
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

    from pydantic import BaseModel, Field

    class ClinicianInsightsResponse(BaseModel):
        insights: list[str] = Field(default_factory=list)

    schema = ClinicianInsightsResponse.model_json_schema()

    prompt = f"""
Return EXACT JSON per schema:
{schema}

You are assisting a clinician. Given the conversation and extracted entities, produce 3-5 concise bullet-point insights that a clinician would find truly helpful. Focus on:
- Possible underlying conditions or differentials to consider
- Medication adherence or confusion issues
- Key lifestyle/diet factors impacting the patient's symptoms
- Next diagnostic steps or clinical considerations

Be specific, interpretive, and medically relevant, not just repetition of the transcript.

--- CONVERSATION ---
{conversation_snippet}
--- ENTITIES ---
{entity_summary}
"""

    insight_result = await handler.process(
        prompts=prompt,
        model="gpt-4o-mini",
        response_type=ClinicianInsightsResponse,
    )

    if insight_result.success and insight_result.data:
        return insight_result.data.insights
    else:
        return []


async def process_conversation(conversation_id: str) -> dict:
    convo_text = read_conversation_text(conversation_id)
    new_hash = get_transcript_hash(convo_text)
    cached_result = load_from_cache(conversation_id)

    if cached_result is not None and cached_result.get("transcript_hash") == new_hash:
        return cached_result

    turns = split_into_turns(convo_text)
    try:
        llm_resp = await run_llm_extraction(conversation_id, turns)
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
        insights = await run_llm_clinician_insights(conversation_id, turns, entities)

        final_result = ConversationAnalysis(
            conversation_id=conversation_id,
            summary=summary,
            entities=entities,
            highlighted_confusions=confusions,
            clinician_insights=insights,
        ).model_dump(mode="json")

        final_result["transcript_hash"] = new_hash
        save_to_cache(conversation_id, final_result)
        return final_result

    except Exception as e:
        print(traceback.format_exc())
        return {
            "conversation_id": conversation_id,
            "error": f"LLM extraction failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }


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
