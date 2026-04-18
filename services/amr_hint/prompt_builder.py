"""
Prompt Builder for AMR Reasoning Data Generation Pipeline.

This module manages all prompts used in the pipeline:
- System prompts for initial AMR generation
- Feedback prompts for error correction loop
- Output format specifications
"""

from typing import List, Dict, Optional
from .AMRHint import AMRHint


class PromptBuilder:
    """Builder class for constructing prompts for AMR reasoning generation."""
    
    # Base system prompt template defining the LLM's role
    SYSTEM_PROMPT_TEMPLATE = """You are an expert in Abstract Meaning Representation (AMR) labeling. Your task is to transform text sentences into AMR graph structures.

## Knowledge
{hints}

**Reasoning Guidelines:**
- Reason naturally using your expert knowledge of framesets and AMR semantics.
- Do NOT reference knowledge as "provided," "given," "according to instructions," or "based on the framesets above."
- Treat all framesets and semantic information as part of your inherent expertise.
- **CRITICAL**: You MUST use the concepts and frames listed in the Knowledge section above when constructing the AMR. These are the correct semantic representations for this specific sentence.
- **CRITICAL**: The FIRST frame/concept listed in Knowledge section MUST BE the root of your AMR graph. Do NOT choose any other concept as root, even if it appears first in the sentence (e.g., discourse markers like "okay-04", "well", etc.). The semantic root is determined by the frame order in Knowledge, not by word order in the sentence.
- For simple sentences: perform quick inference and direct conversion with minimal reasoning.
- For complex sentences: follow some or all of these reasoning steps as needed:

### Identify the Central Concept (Root/Focus)
* The root concept is ALWAYS the FIRST frame/concept in the Knowledge section - do not infer or choose alternatives.
* This root serves as the top-level node of your AMR graph structure.

### Determine Core Roles
* Refer to the semantic frame arguments listed in Knowledge to identify the necessary roles (e.g., `:ARG0`, `:ARG1`, `:ARG2`, etc.).
* Map entities in the sentence to these argument roles.

### Identify Contextual Elements & Attributes (Non-core Roles)
* Consider the presence of auxiliary information such as location (`:location`), time (`:time`), manner (`:manner`), purpose (`:purpose`), or possession (`:poss`),...
* Note that a sentence may contain none or only a few of these attributes depending on the context.

### Assess Logic and Modality
* Check for negation (`:polarity -`) and place it correctly on the negated concept.
* Represent modality factors such as possibility or obligation through their corresponding concepts (`possible-01`, `obligate-01`, etc.) instead of using auxiliary verbs.

### Standardize Entities and Graph Structure
* Format named entities (`person`, `city`, `organization`, etc.) and numbers according to AMR standards.
* Use variables to represent co-reference when an entity appears in multiple positions within the graph.
* Remove non-semantic function words (articles, fillers) to ensure the abstract nature of the representation.

## Output Format
You MUST return JSON with 2 fields:
1. "reasoning": Your reasoning process following the guidelines above. Write naturally without step numbers or labels. Separate major reasoning points with a blank line (double newline). Only include reasoning that is relevant to the sentence complexity.
2. "amr": Complete AMR structure in Penman format
"""

    FEEDBACK_PROMPT_TEMPLATE = """The previous AMR result has errors. Please re-analyze and fix.

## Detected Errors
{errors}

## Correct Items
{correct}

## Previous Output (with errors)
```
{previous_amr}
```

## Requirements
1. Analyze the cause of the errors in the "reasoning" section
2. Fix the errors and create correct AMR in the "amr" section
3. Do not repeat previous mistakes

Return JSON with 2 fields: "reasoning" and "amr"
"""

    USER_PROMPT_TEMPLATE = """Analyze the following sentence and create an AMR structure:

Sentence: {sentence}

Please reason step by step and return JSON with "reasoning" and "amr"."""

    def __init__(self, hint_generator: Optional[AMRHint] = None):
        """
        Initialize PromptBuilder.
        
        Args:
            hint_generator: AMRHint instance for generating hints. 
                          If None, creates a new instance.
        """
        self.hint_generator = hint_generator or AMRHint()
    
    def build_system_prompt(self, gold_amr: str, include_amr_hint: bool = False,
                            hint_percentage: float = 1.0) -> str:
        """
        Build system prompt with hints from gold AMR.
        
        Args:
            gold_amr: Gold standard AMR string for hint extraction
            include_amr_hint: If True, include the gold AMR in the prompt as reference
            hint_percentage: Fraction of hints to include (0.0-1.0). Default 1.0 (all hints).
                           Root frame is always included regardless of percentage.
            
        Returns:
            Formatted system prompt with integrated hints
        """
        # Generate hints based on percentage
        if hint_percentage < 1.0:
            hints = self.hint_generator.get_hints_partial(gold_amr, hint_percentage)
        else:
            hints = self.hint_generator.get_hints(gold_amr)
        
        if not hints:
            hints = "(No special frameset information for this sentence)"
        
        # Add gold AMR as hint if enabled
        if include_amr_hint:
            # Remove wiki tags from AMR before adding to prompt
            cleaned_amr = self.hint_generator.remove_wiki_from_amr(gold_amr)
            hints += f"\n\n### Reference AMR Structure\nUse the following AMR as a reference for your reasoning:\n```\n{cleaned_amr}\n```"
        
        return self.SYSTEM_PROMPT_TEMPLATE.format(hints=hints)
    
    def build_user_prompt(self, sentence: str) -> str:
        """
        Build user prompt with the sentence to analyze.
        
        Args:
            sentence: Input sentence to generate AMR for
            
        Returns:
            Formatted user prompt
        """
        return self.USER_PROMPT_TEMPLATE.format(sentence=sentence)
    
    def build_feedback_prompt(self, 
                              feedback: Dict[str, List],
                              previous_amr: str) -> str:
        """
        Build feedback prompt for error correction.
        
        Args:
            feedback: Dict from AMRFeedback.get_feedback() with 'errors' and 'correct' keys
            previous_amr: The AMR string that had errors
            
        Returns:
            Formatted feedback prompt for re-generation
        """
        # Format errors
        errors_list = feedback.get('errors', [])
        if errors_list:
            error_lines = []
            for i, err in enumerate(errors_list, 1):
                error_lines.append(f"{i}. [{err['type']}] {err['message']}")
                if err.get('details'):
                    error_lines.append(f"   Details: {err['details']}")
            errors_str = "\n".join(error_lines)
        else:
            errors_str = "(No errors)"
        
        # Format correct items
        correct_list = feedback.get('correct', [])
        if correct_list:
            correct_str = "\n".join(f"✓ {item}" for item in correct_list)
        else:
            correct_str = "(Not yet verified)"
        
        return self.FEEDBACK_PROMPT_TEMPLATE.format(
            errors=errors_str,
            correct=correct_str,
            previous_amr=previous_amr
        )
    
    def get_output_schema(self) -> dict:
        """
        Get JSON schema for structured output.
        
        Returns:
            Schema dict compatible with Gemini API response_schema
        """
        return {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning from sentence to AMR structure"
                },
                "amr": {
                    "type": "string", 
                    "description": "Complete AMR structure in Penman format"
                }
            },
            "required": ["reasoning", "amr"]
        }


# Backward compatibility
SYSTEMPROMPT = PromptBuilder.SYSTEM_PROMPT_TEMPLATE