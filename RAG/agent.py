from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Literal, List


class AgentState(TypedDict):
    question: str
    grades: list[str]
    llm_output: str
    documents: list[str]
    on_topic: bool

class GradeQuestion(BaseModel):
    """check whether a question is related to Medical context eg. medical terms, diagnosis, symptoms."""

    score: str = Field(
        description="Question is related to Medical context eg. medical terms, diagnosis, symptoms? If yes -> 'Yes' if not -> 'No'"
    )

class GradeDocuments(BaseModel):
    """Boolean values to check for relevance on retrieved documents."""

    score: str = Field(
        description="Documents are relevant to the question, 'Yes' or 'No'"
    )