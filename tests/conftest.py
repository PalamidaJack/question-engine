import pytest

from qe.runtime.service import BaseService


@pytest.fixture
def mock_llm(monkeypatch):
    async def fake_call(self, model, messages, schema):
        if hasattr(schema, "model_fields") and "claims" in schema.model_fields:
            return schema.model_validate(
                {
                    "claims": [
                        {
                            "subject_entity_id": "K2-18b",
                            "predicate": "has",
                            "object_value": "water vapor",
                            "confidence": 0.88,
                            "reasoning": "extracted in test",
                        }
                    ]
                }
            )

        return schema.model_construct(**{f: f"test_{f}" for f in schema.model_fields})

    monkeypatch.setattr(BaseService, "_call_llm", fake_call)
