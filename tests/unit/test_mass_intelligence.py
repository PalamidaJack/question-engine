from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.services.mass_intelligence.executor import MassIntelligenceExecutor
from qe.services.mass_intelligence.market_agent import ModelMarketAgent
from qe.substrate.model_market import ModelMarketStore


@pytest.fixture
async def store(tmp_path):
    """Create a test database."""
    db_path = str(tmp_path / "test_market.db")
    store = ModelMarketStore(db_path=db_path)
    await store.initialize()
    yield store
    await store.close()


class TestModelMarketStore:
    async def test_add_model(self, store):
        await store.add_or_update_model(
            provider="openrouter",
            model_id="meta-llama/llama-3.3-70b-instruct:free",
            model_name="Llama 3.3 70B",
            context_length=128000,
            capabilities="tools",
            rate_limit_rpm=20,
            rate_limit_rpd=200,
        )

        models = await store.get_available_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "meta-llama/llama-3.3-70b-instruct:free"
        assert models[0]["provider"] == "openrouter"

    async def test_update_existing_model(self, store):
        await store.add_or_update_model(
            provider="openrouter",
            model_id="test/model:free",
            model_name="Test Model",
            context_length=32000,
        )

        await store.add_or_update_model(
            provider="openrouter",
            model_id="test/model:free",
            model_name="Test Model Updated",
            context_length=64000,
        )

        models = await store.get_available_models()
        assert len(models) == 1
        assert models[0]["model_name"] == "Test Model Updated"
        assert models[0]["context_length"] == 64000

    async def test_mark_unavailable(self, store):
        await store.add_or_update_model(
            provider="openrouter",
            model_id="test/model:free",
            model_name="Test Model",
        )

        await store.mark_model_unavailable("openrouter", "test/model:free", "Model removed")

        models = await store.get_available_models()
        assert len(models) == 0

    async def test_record_success(self, store):
        await store.add_or_update_model(
            provider="openrouter",
            model_id="test/model:free",
            model_name="Test Model",
        )

        await store.record_success("openrouter", "test/model:free", 1500.0)

        model = await store.get_model_by_id("openrouter", "test/model:free")
        assert model["success_count"] == 1
        assert model["avg_latency_ms"] == 1500.0

    async def test_record_failure(self, store):
        await store.add_or_update_model(
            provider="openrouter",
            model_id="test/model:free",
            model_name="Test Model",
        )

        await store.record_failure("openrouter", "test/model:free", "Rate limited")

        model = await store.get_model_by_id("openrouter", "test/model:free")
        assert model["failure_count"] == 1
        assert "Rate limited" in model["last_error"]

    async def test_add_error_record(self, store):
        await store.add_error_record(
            provider="openrouter",
            model_id="test/model:free",
            error_code="429",
            error_message="Rate limit exceeded",
            error_type="rate_limit",
        )

        errors = await store.get_unresolved_errors()
        assert len(errors) == 1
        assert errors[0]["error_code"] == "429"
        assert errors[0]["error_type"] == "rate_limit"

    async def test_resolve_error(self, store):
        await store.add_error_record(
            provider="openrouter",
            model_id="test/model:free",
            error_code="429",
            error_message="Rate limit",
            error_type="rate_limit",
        )

        errors = await store.get_unresolved_errors()
        await store.resolve_error(errors[0]["id"], "Backed off successfully")

        resolved = await store.get_unresolved_errors()
        assert len(resolved) == 0

    async def test_provider_crud(self, store):
        await store.add_or_update_provider(
            provider="testprovider",
            api_base="https://api.testprovider.com",
            requires_api_key=True,
            rate_limit_default_rpm=30,
        )

        providers = await store.get_active_providers()
        assert len(providers) == 1
        assert providers[0]["provider"] == "testprovider"
        assert providers[0]["rate_limit_default_rpm"] == 30

    async def test_stats(self, store):
        await store.add_or_update_model(
            provider="openrouter",
            model_id="model1:free",
            model_name="Model 1",
        )
        await store.add_or_update_model(
            provider="openrouter",
            model_id="model2:free",
            model_name="Model 2",
        )

        stats = await store.get_stats()
        assert stats["total_models"] == 2
        assert stats["available_models"] == 2


class TestModelMarketAgent:
    async def test_error_analysis_404(self):
        store = MagicMock()
        agent = ModelMarketAgent(store=store, poll_interval_seconds=60)

        result = agent._analyze_error("404", "Model not found")

        assert result["action"] == "mark_unavailable"

    async def test_error_analysis_rate_limit(self):
        store = MagicMock()
        agent = ModelMarketAgent(store=store, poll_interval_seconds=60)

        result = agent._analyze_error("429", "Rate limit exceeded")

        assert result["action"] == "backoff"

    async def test_error_analysis_timeout(self):
        store = MagicMock()
        agent = ModelMarketAgent(store=store, poll_interval_seconds=60)

        result = agent._analyze_error("timeout", "Request timed out after 30s")

        assert result["action"] == "backoff"

    async def test_error_analysis_server_error(self):
        store = MagicMock()
        agent = ModelMarketAgent(store=store, poll_interval_seconds=60)

        result = agent._analyze_error("500", "Internal server error")

        assert result["action"] == "retry"

    async def test_error_classification(self):
        store = MagicMock()
        agent = ModelMarketAgent(store=store, poll_interval_seconds=60)

        assert agent._classify_error("Rate limit exceeded") == "rate_limit"
        assert agent._classify_error("Request timed out after 30s") == "timeout"
        assert agent._classify_error("Invalid API key") == "authentication"


class TestMassIntelligenceExecutor:
    async def test_execute_no_models(self):
        store = MagicMock()
        store.get_available_models = AsyncMock(return_value=[])

        market_agent = MagicMock()
        executor = MassIntelligenceExecutor(
            store=store,
            market_agent=market_agent,
            default_timeout_seconds=30.0,
        )

        result = await executor.execute("Test prompt")

        assert result.total_models == 0
        assert result.successful == 0

    async def test_extract_error_code(self):
        store = MagicMock()
        market_agent = MagicMock()
        executor = MassIntelligenceExecutor(store=store, market_agent=market_agent)

        error = Exception("404: Model not found")
        code = executor._extract_error_code(error)
        assert code == "404"

        error = Exception("401 Unauthorized")
        code = executor._extract_error_code(error)
        assert code == "401"

        error = Exception("Some random error")
        code = executor._extract_error_code(error)
        assert code == "unknown"


class TestMassIntelligenceResult:
    def test_result_dataclass(self):
        from qe.services.mass_intelligence.executor import ModelResponse

        response = ModelResponse(
            provider="openrouter",
            model_id="test/model:free",
            model_name="Test Model",
            response="Hello world",
            latency_ms=1500.0,
            success=True,
        )

        assert response.provider == "openrouter"
        assert response.success is True
        assert response.response == "Hello world"
