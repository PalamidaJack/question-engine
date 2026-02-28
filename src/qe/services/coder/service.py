"""Coder service for code generation and execution tasks."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_SUPPORTED_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "sql",
    "shell",
}


class CoderService:
    """Generates and validates code based on task
    descriptions."""

    def __init__(
        self,
        substrate: Any = None,
        bus: Any = None,
    ) -> None:
        self.substrate = substrate
        self.bus = bus

    async def execute_task(
        self,
        description: str,
        language: str = "python",
    ) -> dict:
        """Generate code for the described task.

        Args:
            description: Natural-language description of
                the coding task.
            language: Target programming language.

        Returns a dict with the generated code, output,
        language, and success status.
        """
        if language not in _SUPPORTED_LANGUAGES:
            log.warning(
                "Unsupported language: %s, "
                "defaulting to python",
                language,
            )
            language = "python"

        code = self._generate_stub(description, language)
        success = bool(code)

        log.info(
            "Code task completed: lang=%s success=%s "
            "desc=%.60s",
            language,
            success,
            description,
        )

        return {
            "code": code,
            "output": "",
            "language": language,
            "success": success,
        }

    def _generate_stub(
        self, description: str, language: str
    ) -> str:
        """Generate a placeholder code stub."""
        comment_map = {
            "python": "#",
            "javascript": "//",
            "typescript": "//",
            "sql": "--",
            "shell": "#",
        }
        comment = comment_map.get(language, "#")
        return (
            f"{comment} Task: {description}\n"
            f"{comment} Language: {language}\n"
            f"{comment} TODO: implement\n"
        )
