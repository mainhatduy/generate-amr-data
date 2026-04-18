from pydantic import BaseModel

class AMRReasoning(BaseModel):
    amr: str
    sentence: str
    reasoning: str
    
    