from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Literal


class ExtractedEntity(BaseModel):
    type: Literal["condition", "medication", "instruction", "follow_up"]
    name: Optional[str] = None
    dosage: Optional[str] = None
    text: Optional[str] = None
    first_mention_turn: Optional[int] = None
    patient_explanation: Optional[str] = None
    # link removed
    # link: Optional[HttpUrl] = None
    # Add related turns for provenance
    # We'll derive these from first_mention_turn if needed.
    related_turns: List[int] = Field(default_factory=list)
    needs_clarification: bool = False


class AmbiguousTermResolution(BaseModel):
    ambiguous_term: str
    resolved_meaning: str
    reasoning: Optional[str] = None
    turn_id: int


class LLMTermExtractionResponse(BaseModel):
    entities: List[ExtractedEntity]
    context_samples: List[AmbiguousTermResolution] = Field(default_factory=list)


class ExecutiveSummary(BaseModel):
    main_concerns: List[str] = Field(default_factory=list)
    critical_next_steps: List[str] = Field(default_factory=list)


class ConversationAnalysis(BaseModel):
    conversation_id: str
    summary: ExecutiveSummary
    entities: List[ExtractedEntity] = Field(default_factory=list)
    highlighted_confusions: List[str] = Field(default_factory=list)
