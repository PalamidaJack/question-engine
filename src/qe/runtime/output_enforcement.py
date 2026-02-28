"""Structured output enforcement for models that
may not follow instructions."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)


class OutputEnforcer:
    """Ensures structured output from any model."""

    def try_parse_json(
        self, text: str
    ) -> dict | list | None:
        """Attempt to parse JSON from potentially
        messy model output."""
        # Try direct parse first
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON in markdown code blocks
        code_block = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```",
            text,
            re.DOTALL,
        )
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object/array in text
        for pattern in [
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
            r"(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])",
        ]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        return None

    def extract_fields(
        self,
        text: str,
        fields: dict[str, str],
    ) -> dict[str, Any]:
        """Extract fields from text using pattern
        matching.

        fields: {field_name: description}
        """
        result: dict[str, Any] = {}
        for field_name, _desc in fields.items():
            pattern = (
                rf"{field_name}\s*[:=]\s*(.+?)(?:\n|$)"
            )
            match = re.search(
                pattern, text, re.IGNORECASE
            )
            if match:
                value = (
                    match.group(1).strip().strip("\"'")
                )
                result[field_name] = value
        return result

    def repair_json(self, text: str) -> str | None:
        """Attempt to repair common JSON issues."""
        # Remove trailing commas
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        # Add missing quotes around keys
        repaired = re.sub(
            r"(\{|,)\s*(\w+)\s*:",
            r'\1 "\2":',
            repaired,
        )
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return None
