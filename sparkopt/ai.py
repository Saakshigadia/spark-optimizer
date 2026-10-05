"""AI layer: explain bottlenecks in plain language and suggest fixes.

If ANTHROPIC_API_KEY is set, the findings and code are sent to Claude, which
writes an explanation and a rewritten version of the slowest parts. Without a
key, a rule-based summary is produced so the tool still works offline.
"""
import json
import os
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

PROMPT = """You are a Spark performance engineer. A static analyser found these issues in the code below.
Explain the 2-3 biggest bottlenecks in plain language for a junior data engineer, then show the
rewritten code for those parts. Be specific and brief.

Findings (JSON):
{findings}

Code:
```
{code}
```"""


def rule_based_explanation(findings) -> str:
    if not findings:
        return "No common Spark anti-patterns were found. Check the Spark UI for skewed stages and spill if the job is still slow."
    top = findings[:3]
    lines = [f"Found {len(findings)} issue(s). The biggest ones:"]
    for i, f in enumerate(top, 1):
        lines.append(f"{i}. Line {f.line}: {f.title}. {f.why} Fix: {f.fix}")
    highs = sum(f.severity == "high" for f in findings)
    if highs:
        lines.append(f"Start with the {highs} high-severity issue(s); they usually account for most of the runtime.")
    return "\n".join(lines)


def explain(findings, code: str, timeout: int = 30) -> dict:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return {"source": "rules", "text": rule_based_explanation(findings)}
    body = json.dumps({
        "model": DEFAULT_MODEL,
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": PROMPT.format(
            findings=json.dumps([f.to_dict() for f in findings], indent=1), code=code[:12000])}],
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return {"source": "llm", "model": DEFAULT_MODEL, "text": text}
    except Exception as e:  # network down, bad key, rate limit: never break the report
        return {"source": "rules", "text": rule_based_explanation(findings), "llm_error": str(e)}
