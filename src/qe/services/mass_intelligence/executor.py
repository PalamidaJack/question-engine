import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import litellm

from qe.services.mass_intelligence.market_agent import ModelMarketAgent
from qe.substrate.model_market import ModelMarketStore

log = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from a single model."""
    provider: str
    model_id: str
    model_name: str
    response: str
    latency_ms: float
    success: bool
    error: str | None = None
    tokens_used: int = 0


@dataclass
class MassIntelligenceResult:
    """Result from running a prompt across multiple models."""
    prompt: str
    responses: list[ModelResponse]
    total_models: int
    successful: int
    failed: int
    total_time_ms: float


class MassIntelligenceExecutor:
    """
    Service 2: Executes prompts across all available free models in parallel.
    Reports errors back to ModelMarketAgent for analysis.
    """

    def __init__(
        self,
        store: ModelMarketStore,
        market_agent: ModelMarketAgent,
        default_timeout_seconds: float = 30.0,
        max_concurrent: int = 10,
        api_keys: dict[str, str] | None = None,
    ):
        self.store = store
        self.market_agent = market_agent
        self.default_timeout_seconds = default_timeout_seconds
        self.max_concurrent = max_concurrent
        self.api_keys = api_keys or {}
        self._setup_litellm()

    def _setup_litellm(self) -> None:
        """Configure litellm with API keys for various providers."""
        if self.api_keys.get("openrouter"):
            litellm.drop_params = True

        litellm.set_verbose = False

    async def execute(
        self,
        prompt: str,
        system_message: str | None = None,
        max_concurrent: int | None = None,
        timeout_seconds: float | None = None,
        model_ids: list[str] | None = None,
        providers: list[str] | None = None,
    ) -> MassIntelligenceResult:
        """
        Execute a prompt across all available free models.

        Args:
            prompt: The user prompt to send to all models
            system_message: Optional system message
            max_concurrent: Max parallel requests (default from init)
            timeout_seconds: Per-model timeout (default from init)
            model_ids: Optional list of specific model IDs to query
            providers: Optional list of provider names to filter by

        Returns:
            MassIntelligenceResult with all responses
        """
        start_time = time.perf_counter()

        available_models = await self.store.get_available_models()

        if model_ids:
            id_set = set(model_ids)
            available_models = [m for m in available_models if m["model_id"] in id_set]
        if providers:
            prov_set = set(providers)
            available_models = [m for m in available_models if m["provider"] in prov_set]

        if not available_models:
            return MassIntelligenceResult(
                prompt=prompt,
                responses=[],
                total_models=0,
                successful=0,
                failed=0,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        concurrency = max_concurrent or self.max_concurrent
        timeout = timeout_seconds or self.default_timeout_seconds
        semaphore = asyncio.Semaphore(concurrency)

        tasks = [
            self._call_model(
                model,
                prompt,
                system_message,
                semaphore,
                timeout,
            )
            for model in available_models
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = []
        failed = 0

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                model = available_models[i]
                error_response = ModelResponse(
                    provider=model["provider"],
                    model_id=model["model_id"],
                    model_name=model["model_name"],
                    response="",
                    latency_ms=0,
                    success=False,
                    error=str(response),
                )
                valid_responses.append(error_response)
                failed += 1

                await self.market_agent.analyze_and_report_error(
                    provider=model["provider"],
                    model_id=model["model_id"],
                    error_code="exception",
                    error_message=str(response)[:500],
                )
            elif response.success:
                valid_responses.append(response)
            else:
                valid_responses.append(response)
                failed += 1

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return MassIntelligenceResult(
            prompt=prompt,
            responses=valid_responses,
            total_models=len(available_models),
            successful=len(valid_responses) - failed,
            failed=failed,
            total_time_ms=total_time_ms,
        )

    async def _call_model(
        self,
        model: dict[str, Any],
        prompt: str,
        system_message: str | None,
        semaphore: asyncio.Semaphore,
        timeout: float,  # noqa: ASYNC109
    ) -> ModelResponse:
        """Call a single model with timeout and error handling."""
        async with semaphore:
            start_time = time.perf_counter()

            try:
                response = await asyncio.wait_for(
                    self._make_request(model, prompt, system_message),
                    timeout=timeout,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000

                await self.store.record_success(
                    model["provider"],
                    model["model_id"],
                    latency_ms,
                )

                return ModelResponse(
                    provider=model["provider"],
                    model_id=model["model_id"],
                    model_name=model["model_name"],
                    response=response["content"],
                    latency_ms=latency_ms,
                    success=True,
                    tokens_used=response.get("tokens_used", 0),
                )

            except TimeoutError:
                latency_ms = (time.perf_counter() - start_time) * 1000

                await self.market_agent.analyze_and_report_error(
                    provider=model["provider"],
                    model_id=model["model_id"],
                    error_code="timeout",
                    error_message=f"Model timed out after {timeout}s",
                )

                return ModelResponse(
                    provider=model["provider"],
                    model_id=model["model_id"],
                    model_name=model["model_name"],
                    response="",
                    latency_ms=latency_ms,
                    success=False,
                    error=f"Timeout after {timeout}s",
                )

            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                error_message = str(e)
                error_code = self._extract_error_code(e)

                await self.market_agent.analyze_and_report_error(
                    provider=model["provider"],
                    model_id=model["model_id"],
                    error_code=error_code,
                    error_message=error_message[:500],
                )

                return ModelResponse(
                    provider=model["provider"],
                    model_id=model["model_id"],
                    model_name=model["model_name"],
                    response="",
                    latency_ms=latency_ms,
                    success=False,
                    error=error_message,
                )

    async def _make_request(
        self,
        model: dict[str, Any],
        prompt: str,
        system_message: str | None,
    ) -> dict[str, Any]:
        """Make the actual LLM request using litellm."""
        provider = model["provider"]
        model_id = model["model_id"]

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 4096,
        }

        if provider == "openrouter":
            kwargs["model"] = f"openrouter/{model_id}"
            api_key = self.api_keys.get("openrouter") or os.environ.get("OPENROUTER_API_KEY", "")
            if api_key:
                kwargs["api_key"] = api_key

        elif provider == "groq":
            kwargs["model"] = model_id.replace("groq/", "")
            kwargs["api_key"] = self.api_keys.get("groq")

        elif provider == "cerebras":
            kwargs["model"] = model_id.replace("cerebras/", "")
            kwargs["api_key"] = self.api_keys.get("cerebras")

        elif provider == "cohere":
            kwargs["api_key"] = self.api_keys.get("cohere")

        elif provider == "mistral":
            kwargs["api_key"] = self.api_keys.get("mistral")

        elif provider == "google":
            kwargs["model"] = model_id.replace("google/", "")
            kwargs["api_key"] = self.api_keys.get("google")

        elif provider == "hyperbolic":
            kwargs["model"] = model_id.replace("hyperbolic/", "")
            kwargs["api_key"] = self.api_keys.get("hyperbolic")

        elif provider == "sambanova":
            kwargs["model"] = model_id.replace("sambanova/", "")
            kwargs["api_key"] = self.api_keys.get("sambanova")

        elif provider == "scaleway":
            kwargs["model"] = model_id.replace("scaleway/", "")
            kwargs["api_key"] = self.api_keys.get("scaleway")

        elif provider == "cloudflare":
            kwargs["model"] = model_id
            kwargs["api_key"] = self.api_keys.get("cloudflare")

        elif provider == "kilo":
            kwargs["model"] = f"openrouter/{model_id}"
            kwargs["api_base"] = "https://kilo.ai/api/openrouter"
            api_key = self.api_keys.get("kilo") or os.environ.get("KILOCODE_API_KEY", "")
            if api_key:
                kwargs["api_key"] = api_key

        try:
            response = await litellm.acompletion(**kwargs)

            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0

            return {
                "content": content,
                "tokens_used": tokens_used,
            }

        except Exception as e:
            if hasattr(e, "status_code"):
                raise Exception(f"{e.status_code}: {str(e)}") from e
            raise

    def _extract_error_code(self, error: Exception) -> str:
        """Extract error code from exception."""
        error_str = str(error).lower()

        if hasattr(error, "status_code"):
            return str(error.status_code)

        if "401" in error_str or "unauthorized" in error_str:
            return "401"
        if "403" in error_str or "forbidden" in error_str:
            return "403"
        if "404" in error_str or "not found" in error_str:
            return "404"
        if "429" in error_str or "rate limit" in error_str:
            return "429"
        if "500" in error_str or "internal error" in error_str:
            return "500"
        if "502" in error_str or "bad gateway" in error_str:
            return "502"
        if "503" in error_str or "unavailable" in error_str:
            return "503"

        return "unknown"

    async def quick_query(
        self,
        prompt: str,
        max_models: int = 5,
    ) -> MassIntelligenceResult:
        """Quick query with limited models for faster response."""
        available_models = await self.store.get_available_models()
        limited_models = available_models[:max_models]

        concurrency = min(self.max_concurrent, max_models)
        semaphore = asyncio.Semaphore(concurrency)

        start_time = time.perf_counter()

        tasks = [
            self._call_model(model, prompt, None, semaphore, self.default_timeout_seconds)
            for model in limited_models
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = []
        failed = 0

        for response in responses:
            if isinstance(response, Exception):
                failed += 1
            elif response.success:
                valid_responses.append(response)
            else:
                valid_responses.append(response)
                failed += 1

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return MassIntelligenceResult(
            prompt=prompt,
            responses=valid_responses,
            total_models=len(limited_models),
            successful=len(valid_responses) - failed,
            failed=failed,
            total_time_ms=total_time_ms,
        )
