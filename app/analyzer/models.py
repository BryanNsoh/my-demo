from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class ReferencedItem(BaseModel):
    # A generic item referencing certain transcript turns
    title: str
    description: str
    turns: List[int] = Field(default_factory=list)


class PatientFriendlySection(BaseModel):
    # Patient-friendly summary of conditions, instructions, etc.
    main_concerns: List[ReferencedItem] = Field(default_factory=list)
    next_steps: List[ReferencedItem] = Field(default_factory=list)
    medication_notes: List[ReferencedItem] = Field(default_factory=list)
    lifestyle_suggestions: List[ReferencedItem] = Field(default_factory=list)


class ClinicianFriendlySection(BaseModel):
    # Clinician-focused insights: differential diagnoses, recommended tests, ambiguities
    assessment: List[ReferencedItem] = Field(default_factory=list)
    recommendations: List[ReferencedItem] = Field(default_factory=list)
    ambiguities: List[ReferencedItem] = Field(default_factory=list)
    differential_diagnoses: List[ReferencedItem] = Field(default_factory=list)
    diagnostic_steps: List[ReferencedItem] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    type: Literal["condition", "medication", "instruction", "follow_up"]
    name: Optional[str] = None
    dosage: Optional[str] = None
    text: Optional[str] = None
    first_mention_turn: Optional[int] = None
    patient_explanation: Optional[str] = None
    related_turns: List[int] = Field(default_factory=list)
    needs_clarification: bool = False


class AmbiguousTermResolution(BaseModel):
    ambiguous_term: str
    resolved_meaning: str
    reasoning: Optional[str] = None
    turn_id: int


class LLMTermExtractionResponse(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    context_samples: List[AmbiguousTermResolution] = Field(default_factory=list)


class ExecutiveSummary(BaseModel):
    main_concerns: List[str] = Field(default_factory=list)
    critical_next_steps: List[str] = Field(default_factory=list)


class ConversationAnalysis(BaseModel):
    conversation_id: str
    summary: ExecutiveSummary
    entities: List[ExtractedEntity] = Field(default_factory=list)
    highlighted_confusions: List[str] = Field(default_factory=list)
    # Instead of just lists of strings, we have structured patient and clinician info
    patient_friendly: PatientFriendlySection = Field(
        default_factory=PatientFriendlySection
    )
    clinician_friendly: ClinicianFriendlySection = Field(
        default_factory=ClinicianFriendlySection
    )
