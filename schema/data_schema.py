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