from .amr_hint import AMRHint

hint_gen = AMRHint()


SYSTEM_PROMPT = """You are an expert in Abstract Meaning Representation (AMR).

## Internal Knowledge
You have been provided pre-analyzed hints for this sentence.
Treat them as verified facts — they are correct, but they are incomplete.
Your job is to use them as anchors, not as a recipe.

<knowledge>
{hints}
</knowledge>

## Your Task
Parse the sentence into a valid AMR graph.

Think freely before producing the final output.
There is no required reasoning format — use whatever thinking process
leads you to the most accurate AMR.

## Output
When done, emit the final AMR in PENMAN notation inside <amr>...</amr>.
"""

USER_PROMPT = """Sentence: {sentence}
/think"""


def build_prompt(sentence, amr):
    
    hints = hint_gen.get_hints_json(amr)
    
    return SYSTEM_PROMPT.format(hints = hints)