from pydantic import BaseModel
from typing import List


class AMRReasoning(BaseModel):
    amr: str
    sentence: str
    reasoning: str


class SystheticData(BaseModel):
    amr: str
    sentence: str
    model_respose: List[str]


class ReasoningSample(BaseModel):
    """A single scored sample with its thinking process."""
    thinking: str
    pred_amr: str
    f1: float
    precision: float
    recall: float


class DiverseReasoningResult(BaseModel):
    """Result for one input sentence: top-k diverse reasoning paths."""
    id: int | None = None
    sentence: str
    gold_amr: str
    selected_samples: List[ReasoningSample]
    total_generated: int
    best_f1: float
    is_complete: bool