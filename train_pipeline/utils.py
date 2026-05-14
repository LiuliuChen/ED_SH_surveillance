"""
Shared utilities for stage 1 and stage 2 inference scripts.

Mostly response-extraction helpers and JSON parsing recovery.
"""
import json
import re


def extract_segments(resp):
    """Parse an OpenAI Responses API result into (reasoning_text, output_text).

    Splits the response into:
      - reasoning_text: concatenated `reasoning_text` pieces (audit / debug)
      - output_text:    concatenated visible message text (what gets parsed)
    Falls back to `output_text` attribute if the server did not segment.
    """
    rs_parts, msg_parts = [], []

    out = getattr(resp, 'output', None)
    if out:
        for seg in out:
            seg_type = getattr(seg, 'type', '')
            content  = getattr(seg, 'content', []) or []
            if seg_type == 'reasoning':
                for item in content:
                    if getattr(item, 'type', '') == 'reasoning_text':
                        t = getattr(item, 'text', '') or ''
                        if t:
                            rs_parts.append(t)
            elif seg_type == 'message':
                for item in content:
                    if getattr(item, 'type', '') in ('text', 'output_text'):
                        t = getattr(item, 'text', '') or ''
                        if t:
                            msg_parts.append(t)

    if not msg_parts:
        msg = (getattr(resp, 'output_text', None) or '').strip()
        if msg:
            msg_parts.append(msg)

    return ('\n'.join(rs_parts).strip(), '\n'.join(msg_parts).strip())


def safe_parse(text: str):
    """Best-effort JSON parse with fence stripping and first-{...} fallback.

    Returns the parsed dict or None on failure.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        cleaned = re.sub(r'```(?:json)?|```', '', text).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            m = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
    return None
