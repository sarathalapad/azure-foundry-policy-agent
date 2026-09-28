"""Input/output safety screens: offline regex rules and Azure AI Content Safety.

Both screens answer the same question - allow or block, and why - so the agent
applies one policy regardless of provider. The Azure screen fails *closed*: if
the safety service cannot be reached, a public-sector assistant should refuse
rather than answer unscreened.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from .models import SafetyVerdict

log = logging.getLogger(__name__)

SHIELD_API_VERSION = "2024-09-01"

_JAILBREAK = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (your|the|all) (rules|instructions|guidelines|system prompt)",
    r"\b(reveal|print|show|repeat) (me )?(your|the) (system )?(prompt|instructions)\b",
    r"\byou are (now )?(dan|in developer mode|unfiltered|jailbroken)\b",
    r"\b(pretend|act as if) (you have|there are) no (rules|restrictions|filters)\b",
    r"\bdo anything now\b",
]
_HARMFUL = {
    "violence": r"\b(make|build|assemble) (a |an )?(bomb|explosive|weapon)s?\b|\bkill (someone|a person|people)\b",
    "self_harm": r"\b(kill myself|end my life|suicide method|hurt myself)\b",
    "fraud": r"\b(forge|fake|counterfeit) (a |an )?(veloid|passport|identity|id card|document)s?\b",
}
# National ID numbers must never appear in an answer; the corpus contains none, so a match means leakage.
_OUTPUT_PII = {"national_id_number": r"\b\d{3}-\d{3}-\d{3}-\d\b"}


class SafetyScreen(Protocol):
    def check_input(self, text: str) -> SafetyVerdict: ...
    def check_output(self, text: str) -> SafetyVerdict: ...


class RegexSafetyScreen:
    """Deterministic, dependency-free screen used offline and in CI.

    Pattern lists are no substitute for a trained classifier; they exist so the
    blocking path is exercised end to end without cloud access.
    """

    def __init__(self) -> None:
        self._jailbreak = [re.compile(p, re.IGNORECASE) for p in _JAILBREAK]
        self._harmful = {k: re.compile(p, re.IGNORECASE) for k, p in _HARMFUL.items()}
        self._pii = {k: re.compile(p) for k, p in _OUTPUT_PII.items()}

    def _harm(self, text: str) -> list[str]:
        return [f"harmful_content:{name}" for name, rx in self._harmful.items() if rx.search(text)]

    def check_input(self, text: str) -> SafetyVerdict:
        reasons = ["jailbreak_attempt"] if any(rx.search(text) for rx in self._jailbreak) else []
        reasons += self._harm(text)
        return SafetyVerdict.block(*reasons) if reasons else SafetyVerdict.ok()

    def check_output(self, text: str) -> SafetyVerdict:
        reasons = self._harm(text) + [f"pii:{n}" for n, rx in self._pii.items() if rx.search(text)]
        return SafetyVerdict.block(*reasons) if reasons else SafetyVerdict.ok()


class AzureContentSafetyScreen:
    """Prompt Shields (jailbreak) on input, plus harm-category analysis on input and output."""

    def __init__(self, client: Any, block_severity: int = 4):
        self._client, self._threshold = client, block_severity

    def _shield_prompt(self, text: str) -> bool:
        # Prompt Shields is not wrapped by azure-ai-contentsafety 1.0.0, so call the
        # REST operation through the client's authenticated pipeline instead.
        from azure.core.rest import HttpRequest

        request = HttpRequest(
            "POST",
            f"/contentsafety/text:shieldPrompt?api-version={SHIELD_API_VERSION}",
            json={"userPrompt": text, "documents": []},
        )
        response = self._client.send_request(request)
        response.raise_for_status()
        return bool(response.json().get("userPromptAnalysis", {}).get("attackDetected", False))

    def _harm_categories(self, text: str) -> list[str]:
        from azure.ai.contentsafety.models import AnalyzeTextOptions

        result = self._client.analyze_text(AnalyzeTextOptions(text=text[:10000]))
        return [
            f"harmful_content:{str(getattr(item.category, 'value', item.category)).lower()}"
            for item in (result.categories_analysis or [])
            if (item.severity or 0) >= self._threshold
        ]

    def check_input(self, text: str) -> SafetyVerdict:
        try:
            reasons = ["jailbreak_attempt"] if self._shield_prompt(text) else []
            reasons += self._harm_categories(text)
        except Exception as exc:  # any SDK/HTTP failure -> fail closed
            log.error("Content Safety input check failed: %s", type(exc).__name__)
            return SafetyVerdict.block("safety_service_unavailable")
        return SafetyVerdict.block(*reasons) if reasons else SafetyVerdict.ok()

    def check_output(self, text: str) -> SafetyVerdict:
        try:
            reasons = self._harm_categories(text)
        except Exception as exc:
            log.error("Content Safety output check failed: %s", type(exc).__name__)
            return SafetyVerdict.block("safety_service_unavailable")
        return SafetyVerdict.block(*reasons) if reasons else SafetyVerdict.ok()
