from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional


class ExtractedEntity(BaseModel):
    type: str = Field(
        ..., description="Type of entity: condition|medication|instruction|follow_up"
    )
    name: Optional[str] = None
    dosage: Optional[str] = None
    text: Optional[str] = None
    first_mention_sentence: Optional[int] = None


class AmbiguousTermResolution(BaseModel):
    ambiguous_term: str
    resolved_meaning: str
    reasoning: Optional[str] = None
    sentence_id: int


class LLMTermExtractionResponse(BaseModel):
    entities: List[ExtractedEntity]
    context_samples: List[AmbiguousTermResolution] = Field(default_factory=list)


class FinalEntity(BaseModel):
    type: str
    name: Optional[str] = None
    dosage: Optional[str] = None
    text: Optional[str] = None
    first_mention_sentence: Optional[int] = None
    patient_explanation: Optional[str] = None
    link: Optional[HttpUrl] = None


class FinalContextSample(BaseModel):
    ambiguous_term: str
    resolved_meaning: str
    reasoning: Optional[str]
    sentence_id: int


class FinalConversationAnalysis(BaseModel):
    conversation_id: str
    entries: List[FinalEntity] = Field(default_factory=list)
    context_samples: List[FinalContextSample] = Field(default_factory=list)
