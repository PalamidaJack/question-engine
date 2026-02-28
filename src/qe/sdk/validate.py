"""Genome validation for the SDK."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ValidationError:
    """A single validation error."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message

    def __repr__(self) -> str:
        return f"ValidationError({self.field}: {self.message})"


def validate_genome(
    genome_path: str | Path,
) -> list[ValidationError]:
    """Validate a genome TOML file.

    Checks:
    - TOML syntax
    - Required fields present
    - Bus topics exist in protocol
    - System prompt is non-empty
    """
    errors: list[ValidationError] = []
    path = Path(genome_path)

    if not path.exists():
        errors.append(
            ValidationError("path", f"File not found: {path}")
        )
        return errors

    if not path.suffix == ".toml":
        errors.append(
            ValidationError("path", "File must be .toml")
        )

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            errors.append(
                ValidationError(
                    "dependency",
                    "tomllib/tomli not available",
                )
            )
            return errors

    try:
        with open(path, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
    except Exception as exc:
        errors.append(
            ValidationError("syntax", f"TOML parse error: {exc}")
        )
        return errors

    # Check required sections
    if "service" not in data:
        errors.append(
            ValidationError("service", "Missing [service] section")
        )
    else:
        svc = data["service"]
        if "service_id" not in svc:
            errors.append(
                ValidationError(
                    "service.service_id",
                    "Missing service_id",
                )
            )

    # Check for system prompt
    prompts = data.get("prompts", {})
    if not prompts:
        errors.append(
            ValidationError("prompts", "Missing [prompts] section")
        )
    else:
        has_prompt = any(
            isinstance(v, str) and len(v) > 0
            for v in prompts.values()
        )
        if not has_prompt:
            errors.append(
                ValidationError(
                    "prompts",
                    "No non-empty prompt found",
                )
            )

    # Check capabilities
    caps = data.get("capabilities", {})
    if not caps:
        log.debug("validate: no capabilities declared in %s", path)

    # Check bus topics against protocol
    subscribe = caps.get("bus_topics_subscribe", [])
    publish = caps.get("bus_topics_publish", [])
    try:
        from qe.bus.protocol import TOPICS

        for topic in subscribe:
            if topic not in TOPICS and not topic.endswith("*"):
                errors.append(
                    ValidationError(
                        "capabilities.bus_topics_subscribe",
                        f"Unknown topic: {topic}",
                    )
                )
        for topic in publish:
            if topic not in TOPICS and not topic.endswith("*"):
                errors.append(
                    ValidationError(
                        "capabilities.bus_topics_publish",
                        f"Unknown topic: {topic}",
                    )
                )
    except ImportError:
        pass  # Can't validate topics without protocol

    return errors
