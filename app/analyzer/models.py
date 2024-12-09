from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional


class ExtractedEntity(BaseModel):
    type: str = Field(..., description="condition|medication|instruction|follow_up")
    name: Optional[str] = None
    dosage: Optional[str] = None
    text: Optional[str] = None
    first_mention_turn: Optional[int] = Field(
        None, description="The 0-based turn index where it was first mentioned."
    )
    patient_explanation: Optional[str] = Field(
        None, description="A patient-friendly explanation of this entity."
    )
    link: Optional[HttpUrl] = Field(
        None, description="A trusted health resource link about this entity."
    )


class AmbiguousTermResolution(BaseModel):
    ambiguous_term: str
    resolved_meaning: str
    reasoning: Optional[str] = None
    turn_id: int = Field(
        ..., description="Turn index in which the ambiguous term appeared."
    )


class LLMTermExtractionResponse(BaseModel):
    entities: List[ExtractedEntity]
    context_samples: List[AmbiguousTermResolution] = Field(default_factory=list)


class FinalConversationAnalysis(BaseModel):
    conversation_id: str
    entries: List[ExtractedEntity] = Field(default_factory=list)
    context_samples: List[AmbiguousTermResolution] = Field(default_factory=list)
