import asyncio
import logging
from typing import Any

import aiohttp

from qe.substrate.model_market import ModelMarketStore

log = logging.getLogger(__name__)

FREE_MODEL_PROVIDERS = {
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "free_models_endpoint": "/models?free=true",
        "rate_limit_rpm": 20,
        "rate_limit_rpd": 200,
        "model_id_pattern": r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+):free",
    },
    "groq": {
        "api_base": "https://api.groq.com/openai/v1",
        "free_models_endpoint": "/models",
        "rate_limit_rpm": 30,
        "rate_limit_rpd": 14400,
        "requires_api_key": True,
    },
    "cerebras": {
        "api_base": "https://api.cerebras.ai/v1",
        "free_models_endpoint": "/models",
        "rate_limit_rpm": 30,
        "rate_limit_rpd": 14400,
        "requires_api_key": True,
    },
    "cohere": {
        "api_base": "https://api.cohere.ai/v1",
        "free_models_endpoint": "/models",
        "rate_limit_rpm": 20,
        "rate_limit_rpd": 1000,
        "requires_api_key": True,
    },
    "cloudflare": {
        "api_base": "",
        "free_models_endpoint": "",
        "rate_limit_rpm": 100,
        "rate_limit_rpd": 10000,
        "note": "Workers AI - no API key needed for some models",
    },
    "kilo": {
        "api_base": "https://kilo.ai/api/openrouter",
        "free_models_endpoint": "",
        "rate_limit_rpm": 30,
        "rate_limit_rpd": 500,
        "requires_api_key": True,
        "note": "Kilo Code free models via OpenRouter-compatible gateway",
    },
    "mistral": {
        "api_base": "https://api.mistral.ai/v1",
        "free_models_endpoint": "/models",
        "rate_limit_rpm": 30,
        "rate_limit_rpd": 2000,
        "requires_api_key": True,
        "note": "Mistral La Plateforme - free tier available",
    },
    "google": {
        "api_base": "https://generativelanguage.googleapis.com/v1",
        "free_models_endpoint": "/models",
        "rate_limit_rpm": 15,
        "rate_limit_rpd": 1500,
        "requires_api_key": True,
        "note": "Google AI Studio - Gemini 2.5 Flash, Gemma 3 free",
    },
    "hyperbolic": {
        "api_base": "https://api.hyperbolic.xyz/v1",
        "free_models_endpoint": "/models",
        "rate_limit_rpm": 30,
        "rate_limit_rpd": 1000,
        "requires_api_key": True,
        "note": "$1 free credits on signup",
    },
    "sambanova": {
        "api_base": "https://api.sambanova.ai/v1",
        "free_models_endpoint": "/models",
        "rate_limit_rpm": 30,
        "rate_limit_rpd": 1000,
        "requires_api_key": True,
        "note": "$5 free credits for 3 months",
    },
    "scaleway": {
        "api_base": "https://api.scaleway.ai/v1",
        "free_models_endpoint": "/models",
        "rate_limit_rpm": 30,
        "rate_limit_rpd": 5000,
        "requires_api_key": True,
        "note": "1M free tokens",
    },
}


class ModelMarketAgent:
    """
    Service 1: Constantly monitors all LLM providers for free models.
    Discovers new models, tracks rate limits, and analyzes errors from Service 2.
    """

    def __init__(
        self,
        store: ModelMarketStore,
        poll_interval_seconds: int = 900,
        api_keys: dict[str, str] | None = None,
    ):
        self.store = store
        self.poll_interval_seconds = poll_interval_seconds
        self.api_keys = api_keys or {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Start the background monitoring loop."""
        self._running = True
        self._session = aiohttp.ClientSession()
        await self._initialize_providers()
        self._task = asyncio.create_task(self._run_loop())
        log.info("ModelMarketAgent started")

    async def stop(self) -> None:
        """Stop the background monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        log.info("ModelMarketAgent stopped")

    def status(self) -> dict[str, Any]:
        """Return current status of the agent."""
        return {
            "running": self._running,
            "poll_interval_seconds": self.poll_interval_seconds,
        }

    async def _initialize_providers(self) -> None:
        """Initialize known providers in the database."""
        for provider, config in FREE_MODEL_PROVIDERS.items():
            await self.store.add_or_update_provider(
                provider=provider,
                api_base=config.get("api_base", ""),
                requires_api_key=config.get("requires_api_key", True),
                rate_limit_default_rpm=config.get("rate_limit_rpm", 20),
                rate_limit_default_rpd=config.get("rate_limit_rpd", 200),
                notes=config.get("note", ""),
            )

    async def _run_loop(self) -> None:
        """Main background loop that periodically scrapes model lists."""
        while self._running:
            try:
                await self._scrape_all_providers()
                await self._process_error_analysis()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Error in ModelMarketAgent loop: %s", e)

            await asyncio.sleep(self.poll_interval_seconds)

    async def _scrape_all_providers(self) -> None:
        """Scrape model lists from all configured providers."""
        scrape_tasks = [
            self._scrape_openrouter(),
            self._scrape_groq(),
            self._scrape_cerebras(),
            self._scrape_cohere(),
            self._scrape_mistral(),
            self._scrape_google(),
            self._scrape_hyperbolic(),
            self._scrape_sambanova(),
            self._scrape_scaleway(),
            self._scrape_cloudflare_workers(),
        ]

        providers = [
            "openrouter", "groq", "cerebras", "cohere",
            "mistral", "google", "hyperbolic", "sambanova", "scaleway", "cloudflare"
        ]

        results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        for provider, result in zip(providers, results, strict=True):
            if isinstance(result, Exception):
                log.error("Failed to scrape %s: %s", provider, result)

    async def _scrape_openrouter(self) -> None:
        """Scrape free models from OpenRouter."""
        try:
            url = "https://openrouter.ai/api/v1/models?free=true"
            headers = {}
            if self.api_keys.get("openrouter"):
                headers["Authorization"] = f"Bearer {self.api_keys['openrouter']}"

            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("OpenRouter scrape failed: %s", resp.status)
                    return

                data = await resp.json()
                models = data.get("data", [])

                for model in models:
                    model_id = model.get("id", "")
                    if ":free" not in model_id:
                        continue

                    name = model.get("name", model_id)
                    context = model.get("context_length", 32000)

                    capabilities = []
                    if model.get("supports_tools") or model.get("tools"):
                        capabilities.append("tools")
                    if model.get("supports_vision"):
                        capabilities.append("vision")
                    if model.get("supports_reasoning"):
                        capabilities.append("reasoning")

                    await self.store.add_or_update_model(
                        provider="openrouter",
                        model_id=model_id,
                        model_name=name,
                        context_length=context,
                        capabilities=",".join(capabilities),
                        rate_limit_rpm=20,
                        rate_limit_rpd=200,
                    )

                log.info("Scraped %d free models from OpenRouter", len(models))

        except Exception as e:
            log.error("Error scraping OpenRouter: %s", e)

    async def _scrape_groq(self) -> None:
        """Scrape models from Groq."""
        if not self.api_keys.get("groq"):
            log.debug("No Groq API key, skipping")
            return

        try:
            url = "https://api.groq.com/openai/v1/models"
            headers = {"Authorization": f"Bearer {self.api_keys['groq']}"}

            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("Groq scrape failed: %s", resp.status)
                    return

                data = await resp.json()
                models = data.get("data", [])

                for model in models:
                    model_id = model.get("id", "")
                    name = model.get("id", "")
                    context = model.get("context_length", 32000)

                    if not model.get("free", True):
                        continue

                    await self.store.add_or_update_model(
                        provider="groq",
                        model_id=model_id,
                        model_name=name,
                        context_length=context,
                        rate_limit_rpm=30,
                        rate_limit_rpd=14400,
                    )

                log.info("Scraped %d models from Groq", len(models))

        except Exception as e:
            log.error("Error scraping Groq: %s", e)

    async def _scrape_cerebras(self) -> None:
        """Scrape models from Cerebras."""
        if not self.api_keys.get("cerebras"):
            log.debug("No Cerebras API key, skipping")
            return

        try:
            url = "https://api.cerebras.ai/v1/models"
            headers = {"Authorization": f"Bearer {self.api_keys['cerebras']}"}

            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("Cerebras scrape failed: %s", resp.status)
                    return

                data = await resp.json()
                models = data.get("data", [])

                for model in models:
                    model_id = f"cerebras/{model.get('id', '')}"
                    name = model.get("id", "")
                    context = model.get("context_length", 128000)

                    await self.store.add_or_update_model(
                        provider="cerebras",
                        model_id=model_id,
                        model_name=name,
                        context_length=context,
                        rate_limit_rpm=30,
                        rate_limit_rpd=14400,
                    )

                log.info("Scraped %d models from Cerebras", len(models))

        except Exception as e:
            log.error("Error scraping Cerebras: %s", e)

    async def _scrape_cohere(self) -> None:
        """Scrape models from Cohere."""
        if not self.api_keys.get("cohere"):
            log.debug("No Cohere API key, skipping")
            return

        try:
            url = "https://api.cohere.ai/v1/models"
            headers = {"Authorization": f"Bearer {self.api_keys['cohere']}"}

            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("Cohere scrape failed: %s", resp.status)
                    return

                data = await resp.json()
                models = data.get("models", [])

                for model in models:
                    model_id = model.get("name", "")
                    name = model.get("name", "")
                    context = model.get("context_length", 32000)

                    if not model.get("free", False):
                        continue

                    await self.store.add_or_update_model(
                        provider="cohere",
                        model_id=model_id,
                        model_name=name,
                        context_length=context,
                        rate_limit_rpm=20,
                        rate_limit_rpd=1000,
                    )

                log.info("Scraped %d models from Cohere", len(models))

        except Exception as e:
            log.error("Error scraping Cohere: %s", e)

    async def _scrape_mistral(self) -> None:
        """Scrape models from Mistral (La Plateforme)."""
        if not self.api_keys.get("mistral"):
            log.debug("No Mistral API key, skipping")
            return

        try:
            url = "https://api.mistral.ai/v1/models"
            headers = {"Authorization": f"Bearer {self.api_keys['mistral']}"}

            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("Mistral scrape failed: %s", resp.status)
                    return

                data = await resp.json()
                models = data.get("data", [])

                for model in models:
                    model_id = model.get("id", "")
                    name = model.get("id", "")
                    context = model.get("context_length", 32000)

                    await self.store.add_or_update_model(
                        provider="mistral",
                        model_id=model_id,
                        model_name=name,
                        context_length=context,
                        rate_limit_rpm=30,
                        rate_limit_rpd=2000,
                    )

                log.info("Scraped %d models from Mistral", len(models))

        except Exception as e:
            log.error("Error scraping Mistral: %s", e)

    async def _scrape_google(self) -> None:
        """Scrape models from Google AI Studio."""
        if not self.api_keys.get("google"):
            log.debug("No Google API key, skipping")
            return

        try:
            url = "https://generativelanguage.googleapis.com/v1/models"
            headers = {}

            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("Google AI scrape failed: %s", resp.status)
                    return

                data = await resp.json()
                models = data.get("models", [])

                for model in models:
                    model_id = model.get("name", "").replace("models/", "")
                    name = model_id
                    context = model.get("tokenLimit", 32000)

                    await self.store.add_or_update_model(
                        provider="google",
                        model_id=model_id,
                        model_name=name,
                        context_length=context,
                        rate_limit_rpm=15,
                        rate_limit_rpd=1500,
                    )

                log.info("Scraped %d models from Google AI", len(models))

        except Exception as e:
            log.error("Error scraping Google AI: %s", e)

    async def _scrape_hyperbolic(self) -> None:
        """Scrape models from Hyperbolic."""
        if not self.api_keys.get("hyperbolic"):
            log.debug("No Hyperbolic API key, skipping")
            return

        try:
            url = "https://api.hyperbolic.xyz/v1/models"
            headers = {"Authorization": f"Bearer {self.api_keys['hyperbolic']}"}

            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("Hyperbolic scrape failed: %s", resp.status)
                    return

                data = await resp.json()
                models = data.get("data", [])

                for model in models:
                    model_id = model.get("id", "")
                    name = model.get("id", "")
                    context = model.get("context_length", 32000)

                    await self.store.add_or_update_model(
                        provider="hyperbolic",
                        model_id=model_id,
                        model_name=name,
                        context_length=context,
                        rate_limit_rpm=30,
                        rate_limit_rpd=1000,
                    )

                log.info("Scraped %d models from Hyperbolic", len(models))

        except Exception as e:
            log.error("Error scraping Hyperbolic: %s", e)

    async def _scrape_sambanova(self) -> None:
        """Scrape models from SambaNova Cloud."""
        if not self.api_keys.get("sambanova"):
            log.debug("No SambaNova API key, skipping")
            return

        try:
            url = "https://api.sambanova.ai/v1/models"
            headers = {"Authorization": f"Bearer {self.api_keys['sambanova']}"}

            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("SambaNova scrape failed: %s", resp.status)
                    return

                data = await resp.json()
                models = data.get("data", [])

                for model in models:
                    model_id = model.get("id", "")
                    name = model.get("id", "")
                    context = model.get("context_length", 32000)

                    await self.store.add_or_update_model(
                        provider="sambanova",
                        model_id=model_id,
                        model_name=name,
                        context_length=context,
                        rate_limit_rpm=30,
                        rate_limit_rpd=1000,
                    )

                log.info("Scraped %d models from SambaNova", len(models))

        except Exception as e:
            log.error("Error scraping SambaNova: %s", e)

    async def _scrape_scaleway(self) -> None:
        """Scrape models from Scaleway."""
        if not self.api_keys.get("scaleway"):
            log.debug("No Scaleway API key, skipping")
            return

        try:
            url = "https://api.scaleway.ai/v1/models"
            headers = {"Authorization": f"Bearer {self.api_keys['scaleway']}"}

            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("Scaleway scrape failed: %s", resp.status)
                    return

                data = await resp.json()
                models = data.get("data", [])

                for model in models:
                    model_id = model.get("id", "")
                    name = model.get("id", "")
                    context = model.get("context_length", 32000)

                    await self.store.add_or_update_model(
                        provider="scaleway",
                        model_id=model_id,
                        model_name=name,
                        context_length=context,
                        rate_limit_rpm=30,
                        rate_limit_rpd=5000,
                    )

                log.info("Scraped %d models from Scaleway", len(models))

        except Exception as e:
            log.error("Error scraping Scaleway: %s", e)

    async def _scrape_cloudflare_workers(self) -> None:
        """Scrape models from Cloudflare Workers AI."""
        try:
            url = "https://api.cloudflare.com/client/v4/accounts/workers/ai/models"
            headers = {}
            if self.api_keys.get("cloudflare"):
                headers["Authorization"] = f"Bearer {self.api_keys['cloudflare']}"

            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    log.warning("Cloudflare Workers AI scrape failed: %s", resp.status)
                    return

                data = await resp.json()
                models = data.get("result", {}).get("models", [])

                for model in models:
                    model_id = f"@cf/{model.get('id', '')}"
                    name = model.get("name", model_id)
                    context = model.get("limits", {}).get("token_limit", 32000)

                    await self.store.add_or_update_model(
                        provider="cloudflare",
                        model_id=model_id,
                        model_name=name,
                        context_length=context,
                        rate_limit_rpm=100,
                        rate_limit_rpd=10000,
                    )

                log.info("Scraped %d models from Cloudflare Workers AI", len(models))

        except Exception as e:
            log.error("Error scraping Cloudflare Workers AI: %s", e)

    async def _process_error_analysis(self) -> None:
        """Analyze errors from the executor service and update model status."""
        errors = await self.store.get_unresolved_errors()

        for error in errors:
            error_id = error["id"]
            provider = error["provider"]
            model_id = error["model_id"]
            error_code = error["error_code"]
            error_message = error["error_message"]

            resolution = self._analyze_error(error_code, error_message)

            if resolution["action"] == "mark_unavailable":
                await self.store.mark_model_unavailable(
                    provider, model_id, f"{error_code}: {error_message}"
                )
                await self.store.resolve_error(error_id, "Marked unavailable based on error analysis")

            elif resolution["action"] == "backoff":
                await self.store.record_failure(provider, model_id, error_message)
                await self.store.resolve_error(error_id, f"Rate limited - backing off: {resolution['note']}")

            elif resolution["action"] == "retry":
                await self.store.resolve_error(error_id, "Retryable error - will retry on next query")

            else:
                await self.store.resolve_error(error_id, f"No action needed: {resolution['note']}")

    def _analyze_error(self, error_code: str, error_message: str) -> dict[str, str]:
        """Analyze an error and determine what action to take."""
        error_code_lower = str(error_code).lower()
        message_lower = error_message.lower()

        if error_code in ["404", "model_not_found", "not_found"]:
            return {
                "action": "mark_unavailable",
                "note": "Model no longer exists",
            }

        if error_code in ["401", "403", "unauthorized", "forbidden"]:
            return {
                "action": "alert",
                "note": "API key issue - check credentials",
            }

        if error_code in ["429", "rate_limit", "rate_limited"] or "rate limit" in message_lower:
            return {
                "action": "backoff",
                "note": "Reduce request frequency",
            }

        if error_code in ["500", "502", "503", "504", "internal_error", "bad_gateway", "service_unavailable"]:
            return {
                "action": "retry",
                "note": "Server error - will retry",
            }

        if "timeout" in message_lower or "timed out" in message_lower:
            return {
                "action": "backoff",
                "note": "Model is slow - increase timeout",
            }

        if "context length" in message_lower or "max tokens" in message_lower:
            return {
                "action": "mark_unavailable",
                "note": "Model doesn't support requested context",
            }

        return {
            "action": "log",
            "note": "Unknown error type",
        }

    async def analyze_and_report_error(
        self,
        provider: str,
        model_id: str,
        error_code: str | int,
        error_message: str,
    ) -> None:
        """Public method to receive and store errors from the executor."""
        error_type = self._classify_error(error_message)

        await self.store.add_error_record(
            provider=provider,
            model_id=model_id,
            error_code=str(error_code),
            error_message=error_message[:500],
            error_type=error_type,
        )

        log.info(
            "Recorded error for %s/%s: %s (%s)",
            provider,
            model_id,
            error_code,
            error_type,
        )

    def _classify_error(self, error_message: str) -> str:
        """Classify error message into types."""
        message_lower = error_message.lower()

        if "rate limit" in message_lower:
            return "rate_limit"
        if "timeout" in message_lower or "timed out" in message_lower:
            return "timeout"
        if "auth" in message_lower or "api key" in message_lower:
            return "authentication"
        if "not found" in message_lower or "404" in message_lower:
            return "not_found"
        if "invalid" in message_lower or "malformed" in message_lower:
            return "invalid_request"
        if "server" in message_lower or "500" in message_lower:
            return "server_error"

        return "unknown"

    async def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of currently available free models."""
        return await self.store.get_available_models()

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the model market."""
        return await self.store.get_stats()
