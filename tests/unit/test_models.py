import pytest
from pydantic import ValidationError

from qe.models.claim import Claim
from qe.models.envelope import Envelope
from qe.models.genome import Blueprint


def test_envelope_no_args_creates_uuid():
    """Test that an Envelope with no arguments generates a UUID envelope_id."""
    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test-service",
        payload={"text": "test"}
    )
    assert len(envelope.envelope_id) > 0
    assert envelope.schema_version == "1.0"


def test_claim_id_starts_with_clm():
    """Test that a Claim's claim_id starts with 'clm_'."""
    claim = Claim(
        subject_entity_id="entity-123",
        predicate="has-property",
        object_value="value-456",
        confidence=0.9,
        source_service_id="test-service",
        source_envelope_ids=["env-123"]
    )
    assert claim.claim_id.startswith("clm_")
    assert claim.schema_version == "1.0"


def test_blueprint_default_reinforcement_interval():
    """Test that Blueprint's reinforcement_interval_turns defaults to 10."""
    from qe.models.genome import CapabilityDeclaration, ModelPreference

    blueprint = Blueprint(
        service_id="test-service",
        display_name="Test Service",
        version="1.0",
        system_prompt="You are a test service.",
        model_preference=ModelPreference(tier="balanced"),
        capabilities=CapabilityDeclaration()
    )
    assert blueprint.reinforcement_interval_turns == 10


def test_envelope_missing_required_field_raises_validation_error():
    """Test that creating an Envelope without topic raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Envelope(
            source_service_id="test-service",
            payload={"text": "test"}
        )
    # Check that 'topic' is in the error
    errors = exc_info.value.errors()
    assert any("topic" in str(err.get("loc", [])) for err in errors)
