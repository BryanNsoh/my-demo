from pydantic import BaseModel, Field
from typing import List, Optional, Literal

#
# A generic referenced item (e.g., a snippet of advice or an observation)
# that can be tied back to specific turn indices in the transcript.
# Used throughout to provide references in either a patient-friendly
# or clinician-facing summary.
#
class ReferencedItem(BaseModel):
    title: str  # Short title or label for this item
    description: str  # A more detailed explanation or reasoning
    turns: List[int] = Field(default_factory=list)  
    # 'turns' stores the indices of the transcript turns where this item appears or is discussed


#
# Holds patient-facing (i.e., layperson-oriented) pieces of information
# derived from the conversation. Typically includes highlights of main
# concerns, next steps, medication notes, and lifestyle suggestions.
# Each field is a list of ReferencedItems so we can anchor those items back
# to specific transcript turns.
#
class PatientFriendlySection(BaseModel):
    main_concerns: List[ReferencedItem] = Field(default_factory=list)
    next_steps: List[ReferencedItem] = Field(default_factory=list)
    medication_notes: List[ReferencedItem] = Field(default_factory=list)
    lifestyle_suggestions: List[ReferencedItem] = Field(default_factory=list)


#
# Holds clinician-facing insights about the conversation, typically requiring
# more medical/technical language. Includes clinical assessment details,
# recommendations, ambiguities (i.e., unclear information that may require
# follow-up), and potential diagnostic steps.
#
class ClinicianFriendlySection(BaseModel):
    assessment: List[ReferencedItem] = Field(default_factory=list)
    recommendations: List[ReferencedItem] = Field(default_factory=list)
    ambiguities: List[ReferencedItem] = Field(default_factory=list)
    differential_diagnoses: List[ReferencedItem] = Field(default_factory=list)
    diagnostic_steps: List[ReferencedItem] = Field(default_factory=list)


#
# Represents a piece of information the LLM extracts, such as a condition,
# medication, instruction, or follow-up. This object also stores:
# - The text snippet related to that entity
# - The turn where it first appears
# - A simple "patient_explanation" to help non-clinicians understand it
# - "related_turns" pointing to all relevant places in the conversation
# - "needs_clarification" flags if the LLM believes it's ambiguous or incomplete.
#
class ExtractedEntity(BaseModel):
    type: Literal["condition", "medication", "instruction", "follow_up"]
    name: Optional[str] = None
    dosage: Optional[str] = None
    text: Optional[str] = None
    first_mention_turn: Optional[int] = None
    patient_explanation: Optional[str] = None
    related_turns: List[int] = Field(default_factory=list)
    needs_clarification: bool = False


#
# For any term in the transcript that appears ambiguous, we store:
# - The actual ambiguous term
# - The resolved meaning (how the LLM or pipeline interpreted it)
# - Optional reasoning
# - The turn index in which the ambiguity appeared.
#
class AmbiguousTermResolution(BaseModel):
    ambiguous_term: str
    resolved_meaning: str
    reasoning: Optional[str] = None
    turn_id: int


#
# The LLMTermExtractionResponse is the output from an initial LLM pass
# that extracts:
#  - 'entities' (conditions, meds, instructions, etc.)
#  - 'context_samples' which describe how ambiguous terms were resolved.
#
class LLMTermExtractionResponse(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    context_samples: List[AmbiguousTermResolution] = Field(default_factory=list)


#
# A high-level summary intended for quick viewing: 
# - main_concerns: highlights the top issues from the conversation
# - critical_next_steps: outlines major follow-up tasks or instructions
#
class ExecutiveSummary(BaseModel):
    main_concerns: List[str] = Field(default_factory=list)
    critical_next_steps: List[str] = Field(default_factory=list)


#
# The comprehensive output of the entire conversation analysis pipeline.
# - 'conversation_id' identifies which recording/transcript this pertains to
# - 'summary' is an ExecutiveSummary with top-level concerns and actions
# - 'entities' is the list of extracted items from the conversation
# - 'highlighted_confusions' contains text where confusion or uncertainty was detected
# - 'patient_friendly' collects all info needed from the patient’s perspective
# - 'clinician_friendly' collects all info needed from the clinician's perspective
#
class ConversationAnalysis(BaseModel):
    conversation_id: str
    summary: ExecutiveSummary
    entities: List[ExtractedEntity] = Field(default_factory=list)
    highlighted_confusions: List[str] = Field(default_factory=list)
    patient_friendly: PatientFriendlySection = Field(default_factory=PatientFriendlySection)
    clinician_friendly: ClinicianFriendlySection = Field(default_factory=ClinicianFriendlySection)
