from pathlib import Path

import pytest

from qe.kernel.blueprint import load_blueprint
from qe.kernel.registry import ServiceRegistry
from qe.kernel.supervisor import Supervisor
from qe.models.envelope import Envelope


def test_load_blueprint_valid(tmp_path: Path):
    toml_path = tmp_path / "ok.toml"
    toml_path.write_text(
        """
service_id = "researcher_alpha"
display_name = "Researcher Alpha"
version = "1.0"
system_prompt = "Extract claims"

[model_preference]
tier = "balanced"

[capabilities]
bus_topics_subscribe = ["observations.structured"]
bus_topics_publish = ["claims.proposed"]
""".strip()
    )

    bp = load_blueprint(toml_path)
    assert bp.service_id == "researcher_alpha"
    assert bp.capabilities.bus_topics_subscribe == ["observations.structured"]


def test_load_blueprint_missing_required_raises(tmp_path: Path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        """
display_name = "X"
version = "1.0"
system_prompt = "Y"

[model_preference]
tier = "balanced"

[capabilities]
bus_topics_subscribe = ["observations.structured"]
""".strip()
    )

    with pytest.raises(ValueError):
        load_blueprint(bad)


def test_service_registry_get_none_for_missing():
    registry = ServiceRegistry()
    assert registry.get("missing") is None


def test_service_registry_all_services_returns_registered():
    registry = ServiceRegistry()

    class DummyBlueprint:
        service_id = "svc_1"

    class DummyService:
        pass

    bp = DummyBlueprint()
    svc = DummyService()
    registry.register(bp, svc)

    assert registry.get("svc_1") is svc
    assert registry.all_services() == [svc]
    assert registry.all_blueprints() == [bp]


def test_supervisor_loop_detection():
    """Supervisor detects repeated identical publications and triggers circuit break."""
    from qe.bus import get_bus

    bus = get_bus()
    supervisor = Supervisor(bus=bus)

    # Simulate repeated identical publications
    envelope = Envelope(
        topic="claims.proposed",
        source_service_id="researcher_alpha",
        payload={"subject": "test", "predicate": "is", "object": "looping"},
    )

    # Publish enough times to trigger loop detection
    for _ in range(6):
        supervisor._check_loop("researcher_alpha", envelope)

    assert "researcher_alpha" in supervisor._circuit_broken


def test_supervisor_no_false_positive_loop():
    """Different payloads should not trigger loop detection."""
    from qe.bus import get_bus

    bus = get_bus()
    supervisor = Supervisor(bus=bus)

    for i in range(10):
        envelope = Envelope(
            topic="claims.proposed",
            source_service_id="researcher_alpha",
            payload={"subject": "test", "counter": i},
        )
        supervisor._check_loop("researcher_alpha", envelope)

    assert "researcher_alpha" not in supervisor._circuit_broken


def test_supervisor_dynamic_import():
    """service_class entrypoint in genome dynamically loads the service class."""
    from qe.bus.memory_bus import MemoryBus
    from qe.models.genome import Blueprint

    bus = MemoryBus()
    supervisor = Supervisor(bus=bus)

    bp = Blueprint.model_validate({
        "service_id": "custom_svc",
        "display_name": "Custom",
        "version": "1.0",
        "system_prompt": "test",
        "service_class": "qe.services.validator:ClaimValidatorService",
        "model_preference": {"tier": "balanced"},
        "capabilities": {
            "bus_topics_subscribe": ["claims.proposed"],
            "bus_topics_publish": ["claims.committed"],
        },
    })

    svc = supervisor._instantiate_service(bp)
    from qe.services.validator import ClaimValidatorService
    assert isinstance(svc, ClaimValidatorService)


def test_supervisor_prefix_fallback():
    """Without service_class, prefix matching still works."""
    from qe.bus.memory_bus import MemoryBus
    from qe.models.genome import Blueprint

    bus = MemoryBus()
    supervisor = Supervisor(bus=bus)

    bp = Blueprint.model_validate({
        "service_id": "researcher_beta",
        "display_name": "Researcher Beta",
        "version": "1.0",
        "system_prompt": "test",
        "model_preference": {"tier": "balanced"},
        "capabilities": {
            "bus_topics_subscribe": ["observations.structured"],
            "bus_topics_publish": ["claims.proposed"],
        },
    })

    svc = supervisor._instantiate_service(bp)
    from qe.services.researcher import ResearcherService
    assert isinstance(svc, ResearcherService)


def test_load_blueprint_with_constitution(tmp_path: Path):
    """Blueprint loads constitution field from genome TOML."""
    toml_path = tmp_path / "with_constitution.toml"
    toml_path.write_text(
        """
service_id = "researcher_alpha"
display_name = "Researcher Alpha"
version = "1.0"
system_prompt = "Extract claims"
constitution = "Never fabricate evidence."

[model_preference]
tier = "balanced"

[capabilities]
bus_topics_subscribe = ["observations.structured"]
bus_topics_publish = ["claims.proposed"]
""".strip()
    )

    bp = load_blueprint(toml_path)
    assert bp.constitution == "Never fabricate evidence."
