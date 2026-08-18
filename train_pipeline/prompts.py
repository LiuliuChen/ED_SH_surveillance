"""
Prompt strings and prompt builders for Stage 1 (zero-shot screening) and
Stage 2 (CoT structured extraction).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import STAGE2


# =============================================================================
# Stage 1: zero-shot screening prompt (binary self-harm decision)
# =============================================================================
STAGE1_SYSTEM = """
    You are a helpful clinical assistant trained to determine whether a triage note is SELF-HARM-RELATED.

    Count as "yes" (even if intent is unclear/benign):
    - Ingestion/overuse/overdose ("OD")/mixing of alcohol (“ETOH”), illicit drugs, prescription meds (e.g., diazepam/benzos), OTC meds, or supplements (e.g., sleep aids).
    - Taking “extra”, “multiple”, “x tablets/capsules”, “double dose”, “handful” compared to usual dosage.
    - Self-harm methods: cutting, ligature/strangulation, head-banging, burning, jumping, poisoning, inserting objects.
    - If ambiguous but ingestion/overuse is plausible from wording, prefer "yes".
    - Suicidal ideation/plan/attempt/threat/act (even if later denied).
    - Highly implausible accident patterns (e.g., multiple lacerations; cuts to wrist/neck) suggest self-infliction.

    Count as "no":
    - If medication GIVEN/ADMINISTERED by staff/paramedics only.
    - If resulted from assault by another person.
    - If no physical symptoms present (e.g., "anxious").
    - Clearly accidental mechanism with no self-harm context (e.g., single kitchen cut while cooking).

    Count as "unsure":
    - If there is insufficient information to tell.

    Output must be valid JSON only. No extra text. Only reasoning internally.

    Expected JSON:
      {"self-harm-related":"yes"|"no"|"unsure", "reason":"<1-20 words short explanation>"}
    """


def build_stage1_prompt(note: str) -> tuple[str, str]:
    """Return (system_instruction, user_prompt) for one triage note."""
    user_prompt = f"""
    Task: Classify this triage note. Only produce the final JSON.

    Output JSON only:{{"self-harm-related": "yes"|"no"|"unsure", "reason": "<1-20 words, short explanation>"}}

   Note: "{note}"
   """.strip()
    return STAGE1_SYSTEM, user_prompt


# =============================================================================
# Stage 2: CoT structured extraction prompt
# =============================================================================
STAGE2_SYSTEM = """
You are a helpful clinical assistant trained to determine whether a triage note presents evidence of self-harm.

IMPORTANT GUIDELINES:
**Self harm** = *Any intentional act undertaken by a person that results in injury or damage to that person’s own 
body, regardless of the person’s motivation (e.g., to relieve emotional distress, to feel control, to communicate 
distress, or as a precursor to a suicidal act). The act may or may not be accompanied by an intent to die.*

Key points:
- **Intentional**: The person deliberately initiates the act. Accidental injury is **not** self harm.
- **Act on one’s own body**: Includes cutting, burning, hitting, scratching, overdosing, ingesting harmful substances, inserting objects, or any other method that causes physical injury.
- **Resulting in injury or damage**: The act produces a visible wound, physiological change, or a medical risk (e.g., a superficial cut, an overdose, or a self inflicted fracture).
- **Motivation not limited to suicide**: Self harm can be **non suicidal self injury (NSSI)** (i.e., no intent to die) *or* can be part of a **suicidal** act. Both fall under the umbrella term “self harm.”
- **Any degree of severity**: From minor scratches to life threatening injuries are included.

Clarifications:
- Overdosing (medications/substances): Any ingestion (greater than prescribed/expected), unknown “large” quantity, or mixing multiple substances (including alcohol) in a way that creates medical risk.
- Self poisoning (non medicinal toxins): Ingestion/exposure to substances like bleach, pesticides, cleaners, etc.
- In the presence of suicidal ideation or stated distress, multi tablet ingestion or mixing substances generally implies self inflicted and intentional injury unless the note clearly states accidental injury.
- Thoughts of self harm, historical self harm, planned self‑harm without a recent act, or accidental overdoses/poisoning should not be counted as self‑harm.
"""


def build_stage2_prompt(note: str) -> list[dict]:
    """Return chat messages list for Stage 2 CoT extraction on one note."""
    method_labels = '|'.join(STAGE2['method_labels'])
    timing_labels = '|'.join(STAGE2['timing_labels'])
    intent_labels = '|'.join(STAGE2['intentionality_labels'])
    final_labels  = '|'.join(STAGE2['final_labels'])

    user_content = f"""Triage note: {note}

Think step-by-step internally. Only return a SINGLE JSON object (no prose, no markdown, no code fences) with EXACTLY this schema and label sets:

{{
  "step0": {{"label": "Keyword Cues",   "value": "<Yes|No>",             "evidence": "<short quote from note or None>", "reasoning": "<verbatim self-harm phrases or None>"}},
  "step1": {{"label": "Act on body",    "value": "<Yes|No>",             "evidence": "<short quote or None>",           "reasoning": "why this is/isn't an act on the body"}},
  "step2": {{"label": "Injury",         "value": "<Yes|No>",             "evidence": "<short quote or None>",           "reasoning": "injury or medical risk present?"}},
  "step3": {{"label": "Method",         "value": "<{method_labels}>",    "evidence": "<short quote or None>",           "reasoning": "1-line detail"}},
  "step4": {{"label": "Timing",         "value": "<{timing_labels}>",    "evidence": "<short quote or None>",           "reasoning": "When did the act occur relative to this presentation?"}},
  "step5": {{"label": "Intentionality", "value": "<{intent_labels}>",    "evidence": "<short quote or None>",           "reasoning": "why"}},
  "summary":        {{"label": "<{final_labels}>", "evidence": "<single strongest quote or None>", "reasoning": "brief reasoning"}},
  "final_decision": "<{final_labels}>"
}}
"""
    return [
        {'role': 'system', 'content': STAGE2_SYSTEM},
        {'role': 'user',   'content': user_content},
    ]
