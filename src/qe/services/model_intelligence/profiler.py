"""Benchmark runner for model profiling."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import litellm

log = logging.getLogger(__name__)


# -- Benchmark definitions per category ------------------------------------

_BENCHMARKS: dict[str, list[dict[str, Any]]] = {
    # ── Tier 1: automated scoring ────────────────────────────────
    "structured_output": [
        {
            "name": "json_object",
            "prompt": (
                "Return a JSON object with exactly these keys: "
                '"name" (string), "age" (integer), "hobbies" (array of strings). '
                "Describe a fictional person. Output ONLY valid JSON."
            ),
            "scorer": "json_schema",
            "expected_keys": ["name", "age", "hobbies"],
        },
        {
            "name": "json_array",
            "prompt": (
                "Return a JSON array of 3 objects, each with keys "
                '"city" and "population". Output ONLY valid JSON.'
            ),
            "scorer": "json_array",
        },
        {
            "name": "nested_json",
            "prompt": (
                "Return a JSON object describing a book with keys: "
                '"title", "author", and "chapters" (an array of objects '
                'each with "number" and "title"). Output ONLY valid JSON.'
            ),
            "scorer": "json_schema",
            "expected_keys": ["title", "author", "chapters"],
        },
    ],
    "tool_calling": [
        {
            "name": "single_tool_selection",
            "prompt": "What is the weather in San Francisco today?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name",
                                },
                            },
                            "required": ["location"],
                        },
                    },
                },
            ],
            "expected_tool": "get_weather",
            "scorer": "tool_call",
        },
        {
            "name": "tool_with_params",
            "prompt": "Convert 100 USD to EUR.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "convert_currency",
                        "description": "Convert between currencies",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "amount": {"type": "number"},
                                "from_currency": {"type": "string"},
                                "to_currency": {"type": "string"},
                            },
                            "required": [
                                "amount",
                                "from_currency",
                                "to_currency",
                            ],
                        },
                    },
                },
            ],
            "expected_tool": "convert_currency",
            "scorer": "tool_call",
        },
    ],
    "factual_accuracy": [
        {
            "name": "capital_city",
            "prompt": (
                "What is the capital of Japan? "
                "Reply with only the city name."
            ),
            "expected": "Tokyo",
            "scorer": "contains",
        },
        {
            "name": "element_symbol",
            "prompt": (
                "What is the chemical symbol for gold? "
                "Reply with only the symbol."
            ),
            "expected": "Au",
            "scorer": "contains",
        },
        {
            "name": "planet_count",
            "prompt": (
                "How many planets are in our solar system? "
                "Reply with only the number."
            ),
            "expected": "8",
            "scorer": "contains",
        },
    ],
    "code_generation": [
        {
            "name": "fizzbuzz",
            "prompt": (
                "Write a Python function called fizzbuzz(n) that returns "
                '"FizzBuzz" if n is divisible by both 3 and 5, "Fizz" if '
                'divisible by 3, "Buzz" if divisible by 5, and str(n) '
                "otherwise. Output ONLY the function definition."
            ),
            "scorer": "code_keywords",
            "required_keywords": ["def fizzbuzz", "FizzBuzz", "Fizz", "Buzz"],
        },
        {
            "name": "reverse_string",
            "prompt": (
                "Write a Python function called reverse_string(s) that "
                "returns the reversed string. Do not use slicing. "
                "Output ONLY the function definition."
            ),
            "scorer": "code_keywords",
            "required_keywords": ["def reverse_string"],
        },
        {
            "name": "factorial",
            "prompt": (
                "Write a Python function called factorial(n) that computes "
                "n! recursively. Include a base case for n <= 1. "
                "Output ONLY the function definition."
            ),
            "scorer": "code_keywords",
            "required_keywords": ["def factorial", "return"],
        },
    ],
    "instruction_following": [
        {
            "name": "exact_count",
            "prompt": (
                "List exactly 5 colors, one per line. "
                "Do not include numbers or bullets."
            ),
            "scorer": "line_count",
            "expected_lines": 5,
        },
        {
            "name": "format_constraint",
            "prompt": (
                "Write exactly 3 words about the ocean. "
                "Do not use punctuation."
            ),
            "scorer": "word_count",
            "expected_words": 3,
        },
        {
            "name": "multi_constraint",
            "prompt": (
                "Write a sentence about dogs that: "
                "1) starts with the word 'The', "
                "2) contains exactly 10 words, "
                "3) ends with an exclamation mark."
            ),
            "scorer": "multi_constraint",
        },
    ],
    "math_logic": [
        {
            "name": "arithmetic",
            "prompt": "What is 347 + 258? Reply with only the number.",
            "expected": "605",
            "scorer": "contains",
        },
        {
            "name": "percentage",
            "prompt": "What is 15% of 200? Reply with only the number.",
            "expected": "30",
            "scorer": "contains",
        },
        {
            "name": "deduction",
            "prompt": (
                "If all roses are flowers and some flowers fade quickly, "
                "can we conclude that some roses fade quickly? "
                "Answer only 'yes' or 'no'."
            ),
            "expected": "no",
            "scorer": "contains",
        },
    ],
    "reasoning": [
        {
            "name": "causal",
            "prompt": (
                "A farmer has 17 sheep. All but 9 die. "
                "How many sheep are left? Reply with only the number."
            ),
            "expected": "9",
            "scorer": "contains",
        },
        {
            "name": "sequence",
            "prompt": (
                "What is the next number in the sequence: "
                "2, 4, 8, 16, __? Reply with only the number."
            ),
            "expected": "32",
            "scorer": "contains",
        },
        {
            "name": "analogy",
            "prompt": (
                "Complete the analogy: Hot is to cold as up is to __. "
                "Reply with only one word."
            ),
            "expected": "down",
            "scorer": "contains_ci",
        },
    ],
    "summarization": [
        {
            "name": "paragraph_summary",
            "prompt": (
                "Summarize the following in one sentence: "
                "'Photosynthesis is the process by which green plants "
                "and some other organisms use sunlight to synthesize "
                "nutrients from carbon dioxide and water. Photosynthesis "
                "in plants generally involves the green pigment "
                "chlorophyll and generates oxygen as a by-product.'"
            ),
            "scorer": "summary_quality",
            "key_terms": [
                "photosynthesis", "plants", "sunlight", "oxygen",
            ],
        },
        {
            "name": "key_points",
            "prompt": (
                "Extract 3 key points from this text: "
                "'The internet has revolutionized communication, enabling "
                "instant messaging across the globe. It has also "
                "transformed commerce through e-commerce platforms. "
                "Additionally, it has democratized access to information "
                "through search engines and online encyclopedias.'"
            ),
            "scorer": "summary_quality",
            "key_terms": [
                "communication", "commerce", "information",
            ],
        },
    ],
    "creative": [
        {
            "name": "haiku",
            "prompt": (
                "Write a haiku about the moon. "
                "Format: three lines with 5-7-5 syllable pattern."
            ),
            "scorer": "creative_structure",
            "expected_lines": 3,
        },
        {
            "name": "metaphor",
            "prompt": (
                "Write a single creative metaphor comparing "
                "time to a river. One sentence only."
            ),
            "scorer": "creative_length",
            "min_words": 5,
        },
    ],
}


class ModelProfiler:
    """Runs a suite of benchmarks against a model and returns scored results.

    Tier 1 benchmarks (structured_output, tool_calling, factual_accuracy,
    code_generation, instruction_following, math_logic, reasoning) are
    scored programmatically.

    Tier 2 benchmarks (summarization, creative) store raw responses for
    later subjective analysis and use heuristic scoring.
    """

    BENCHMARK_CATEGORIES: list[str] = list(_BENCHMARKS.keys())

    async def run_benchmarks(
        self, model_id: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run all benchmark categories against *model_id*.

        Returns ``{category: [result_dict, ...]}`` where each result_dict
        contains: category, name, score, latency_ms, raw.
        """
        results: dict[str, list[dict[str, Any]]] = {}
        for category, benchmarks in _BENCHMARKS.items():
            cat_results: list[dict[str, Any]] = []
            for bench in benchmarks:
                result = await self._run_single(model_id, category, bench)
                cat_results.append(result)
            results[category] = cat_results
        return results

    # ------------------------------------------------------------------
    # Single benchmark execution
    # ------------------------------------------------------------------

    async def _run_single(
        self,
        model_id: str,
        category: str,
        bench: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one benchmark and return a scored result dict."""
        name = bench["name"]
        prompt = bench["prompt"]
        tools = bench.get("tools")
        scorer = bench.get("scorer", "passthrough")

        t0 = time.monotonic()
        raw = ""
        tool_calls_data: list[dict[str, Any]] = []

        try:
            kwargs: dict[str, Any] = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "timeout": 60,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await litellm.acompletion(**kwargs)
            latency_ms = int((time.monotonic() - t0) * 1000)

            choice = response.choices[0]
            raw = choice.message.content or ""
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                tool_calls_data = [
                    {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in choice.message.tool_calls
                ]
        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log.debug(
                "profiler.benchmark_failed model=%s bench=%s err=%s",
                model_id,
                name,
                exc,
            )
            return {
                "category": category,
                "name": name,
                "score": 0.0,
                "latency_ms": latency_ms,
                "raw": f"ERROR: {exc}",
            }

        score = self._score(scorer, raw, bench, tool_calls_data)
        return {
            "category": category,
            "name": name,
            "score": round(score, 4),
            "latency_ms": latency_ms,
            "raw": raw[:2000],
        }

    # ------------------------------------------------------------------
    # Scoring strategies
    # ------------------------------------------------------------------

    def _score(
        self,
        scorer: str,
        raw: str,
        bench: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> float:
        """Dispatch to the appropriate scoring function."""
        if scorer == "json_schema":
            return self._score_json_schema(raw, bench)
        if scorer == "json_array":
            return self._score_json_array(raw)
        if scorer == "tool_call":
            return self._score_tool_call(tool_calls, bench)
        if scorer == "contains":
            return self._score_contains(raw, bench["expected"])
        if scorer == "contains_ci":
            return self._score_contains_ci(raw, bench["expected"])
        if scorer == "code_keywords":
            return self._score_code_keywords(raw, bench)
        if scorer == "line_count":
            return self._score_line_count(raw, bench)
        if scorer == "word_count":
            return self._score_word_count(raw, bench)
        if scorer == "multi_constraint":
            return self._score_multi_constraint(raw)
        if scorer == "summary_quality":
            return self._score_summary(raw, bench)
        if scorer == "creative_structure":
            return self._score_creative_structure(raw, bench)
        if scorer == "creative_length":
            return self._score_creative_length(raw, bench)
        # passthrough — non-zero if any content was returned
        return 0.5 if raw.strip() else 0.0

    # ── JSON scoring ─────────────────────────────────────────────

    @staticmethod
    def _score_json_schema(raw: str, bench: dict) -> float:
        """Score JSON output for schema compliance."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [
                ln for ln in lines if not ln.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return 0.0
        if not isinstance(data, dict):
            return 0.2
        expected_keys = bench.get("expected_keys", [])
        if expected_keys:
            present = sum(1 for k in expected_keys if k in data)
            return present / len(expected_keys)
        return 1.0

    @staticmethod
    def _score_json_array(raw: str) -> float:
        """Score JSON array output."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [
                ln for ln in lines if not ln.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return 0.0
        if not isinstance(data, list):
            return 0.2
        if len(data) >= 3:
            return 1.0
        return len(data) / 3.0

    # ── Tool-call scoring ────────────────────────────────────────

    @staticmethod
    def _score_tool_call(
        tool_calls: list[dict[str, Any]], bench: dict,
    ) -> float:
        """Score whether the model selected the correct tool."""
        expected = bench.get("expected_tool", "")
        if not tool_calls:
            return 0.0
        for tc in tool_calls:
            if tc.get("name") == expected:
                # Validate arguments are parseable JSON
                try:
                    json.loads(tc.get("arguments", "{}"))
                    return 1.0
                except (json.JSONDecodeError, ValueError):
                    return 0.5
        return 0.0

    # ── Text matching ──────────────────────────────────────────────

    @staticmethod
    def _score_contains(raw: str, expected: str) -> float:
        return 1.0 if expected in raw else 0.0

    @staticmethod
    def _score_contains_ci(raw: str, expected: str) -> float:
        return 1.0 if expected.lower() in raw.lower() else 0.0

    # ── Code scoring ─────────────────────────────────────────────

    @staticmethod
    def _score_code_keywords(raw: str, bench: dict) -> float:
        """Score code generation by keyword presence."""
        keywords = bench.get("required_keywords", [])
        if not keywords:
            return 0.5 if raw.strip() else 0.0
        found = sum(1 for kw in keywords if kw in raw)
        return found / len(keywords)

    # ── Instruction-following scoring ──────────────────────────────

    @staticmethod
    def _score_line_count(raw: str, bench: dict) -> float:
        """Score based on matching expected line count."""
        expected = bench.get("expected_lines", 0)
        lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
        if not expected:
            return 0.5
        return 1.0 if len(lines) == expected else max(0.0, 1.0 - abs(len(lines) - expected) * 0.2)

    @staticmethod
    def _score_word_count(raw: str, bench: dict) -> float:
        """Score based on matching expected word count."""
        expected = bench.get("expected_words", 0)
        words = raw.strip().split()
        if not expected:
            return 0.5
        return 1.0 if len(words) == expected else max(0.0, 1.0 - abs(len(words) - expected) * 0.2)

    @staticmethod
    def _score_multi_constraint(raw: str) -> float:
        """Score a response with multiple constraints."""
        score = 0.0
        text = raw.strip()
        # Constraint 1: starts with 'The'
        if text.startswith("The"):
            score += 0.33
        # Constraint 2: approximately 10 words
        words = text.split()
        if 9 <= len(words) <= 11:
            score += 0.34
        # Constraint 3: ends with '!'
        if text.endswith("!"):
            score += 0.33
        return round(score, 2)

    # ── Summary scoring ──────────────────────────────────────────

    @staticmethod
    def _score_summary(raw: str, bench: dict) -> float:
        """Score summary quality by key term coverage."""
        key_terms = bench.get("key_terms", [])
        if not key_terms:
            return 0.5 if raw.strip() else 0.0
        lower = raw.lower()
        found = sum(1 for t in key_terms if t.lower() in lower)
        return found / len(key_terms)

    # ── Creative scoring ─────────────────────────────────────────

    @staticmethod
    def _score_creative_structure(raw: str, bench: dict) -> float:
        """Score creative output by structural expectations."""
        expected_lines = bench.get("expected_lines", 3)
        lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
        if len(lines) == expected_lines:
            return 1.0
        if lines:
            return 0.5
        return 0.0

    @staticmethod
    def _score_creative_length(raw: str, bench: dict) -> float:
        """Score creative output by minimum word count."""
        min_words = bench.get("min_words", 5)
        words = raw.strip().split()
        if len(words) >= min_words:
            return 1.0
        if words:
            return len(words) / min_words
        return 0.0
