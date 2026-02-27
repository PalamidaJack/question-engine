from pathlib import Path

import pytest

from qe.kernel.blueprint import load_blueprint
from qe.kernel.registry import ServiceRegistry


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
