"""Tests for Phases 13-16: H-Neurons, Advanced Intelligence, Multimodal, Channels."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.channels.base import ChannelAdapter
from qe.channels.notifications import (
    NotificationPreferences,
    NotificationPriority,
    NotificationRouter,
)
from qe.channels.webhook import WebhookAdapter
from qe.ingest.documents import DocumentChunk, DocumentIngestor
from qe.ingest.ocr import OCRProcessor, OCRResult
from qe.ingest.voice import TranscriptSegment, VoiceIngestor, VoiceObservation
from qe.runtime.conformal import ConformalPredictor, PredictionSet
from qe.runtime.h_neuron_integration import HNeuronIntegration
from qe.runtime.h_neurons import HNeuronProfile
from qe.runtime.self_correction import SelfCorrectionEngine
from qe.runtime.users import UserManager
from qe.substrate.inference import (
    AggregateTemplate,
    SymbolicInferenceEngine,
    TemporalTemplate,
    TransitiveTemplate,
)

# ── Phase 13: H-Neuron Integration ────────────────────────────────


class TestHNeuronProfile:
    def test_profile_creation(self):
        profile = HNeuronProfile(
            model_name="test-model",
            neurons_by_layer={0: [1, 2, 3], 5: [10, 20]},
        )
        assert profile.model_name == "test-model"
        assert profile.total_neurons == 5
        assert profile.calibration_size == 0

    def test_profile_save_and_load(self):
        from qe.runtime.h_neurons import HNeuronProfiler

        tmp = tempfile.mkdtemp()
        profiler = HNeuronProfiler(tmp)

        profile = HNeuronProfile(
            model_name="test-model",
            neurons_by_layer={0: [1, 2], 3: [5]},
            calibration_size=100,
        )
        # Use default path (profile_dir/<safe_name>.json)
        saved_path = profiler.save_profile(profile)
        assert saved_path.exists()

        loaded = profiler.load_profile("test-model")
        assert loaded is not None
        assert loaded.model_name == "test-model"
        assert loaded.neurons_by_layer == {0: [1, 2], 3: [5]}
        assert loaded.total_neurons == 3

    def test_list_profiles(self):
        from qe.runtime.h_neurons import HNeuronProfiler

        tmp = tempfile.mkdtemp()
        profiler = HNeuronProfiler(tmp)

        profile = HNeuronProfile(
            model_name="model-a",
            neurons_by_layer={0: [1]},
        )
        profiler.save_profile(profile, Path(tmp) / "model_a.json")

        profiles = profiler.list_profiles()
        assert len(profiles) == 1
        assert "model-a" in profiles


class TestHNeuronIntegration:
    def test_risk_adjusted_profile_high(self):
        integration = HNeuronIntegration(profile_dir=tempfile.mkdtemp())
        assert integration.get_risk_adjusted_profile(0.8) == "high_risk"

    def test_risk_adjusted_profile_moderate(self):
        integration = HNeuronIntegration(profile_dir=tempfile.mkdtemp())
        assert integration.get_risk_adjusted_profile(0.5) == "moderate_risk"

    def test_risk_adjusted_profile_standard(self):
        integration = HNeuronIntegration(profile_dir=tempfile.mkdtemp())
        assert integration.get_risk_adjusted_profile(0.2) == "standard"

    @pytest.mark.asyncio
    async def test_enhance_verification(self):
        integration = HNeuronIntegration(profile_dir=tempfile.mkdtemp())
        result = {"confidence": 0.9, "output": "test"}
        enhanced = await integration.enhance_verification(result, 0.8)
        assert "h_neuron_risk" in enhanced
        assert enhanced["verification_profile"] == "high_risk"
        assert enhanced["confidence"] < 0.9
        assert "h_neuron_high_risk" in enhanced.get("flags", [])

    @pytest.mark.asyncio
    async def test_enhance_verification_low_risk(self):
        integration = HNeuronIntegration(profile_dir=tempfile.mkdtemp())
        result = {"confidence": 0.9}
        enhanced = await integration.enhance_verification(result, 0.1)
        assert enhanced["verification_profile"] == "standard"
        assert enhanced["confidence"] > 0.85

    def test_calibration_features(self):
        integration = HNeuronIntegration(profile_dir=tempfile.mkdtemp())
        features = integration.get_calibration_features(0.75)
        assert features["h_neuron_risk"] == 0.75
        assert features["h_neuron_is_high_risk"] is True
        assert features["h_neuron_available"] is True
        assert 0 <= features["h_neuron_risk_bucket"] <= 9

    def test_is_available_without_torch(self):
        integration = HNeuronIntegration(profile_dir=tempfile.mkdtemp())
        # Without torch installed (in test env), should return False
        # or True depending on environment; just check it doesn't crash
        result = integration.is_available()
        assert isinstance(result, bool)


# ── Phase 14: Conformal Prediction ────────────────────────────────


class TestConformalPredictor:
    def test_calibrate(self):
        cp = ConformalPredictor(coverage=0.90)
        data = [(0.9, True)] * 90 + [(0.8, False)] * 10
        cp.calibrate(data)
        assert cp.is_calibrated()

    def test_not_calibrated(self):
        cp = ConformalPredictor()
        assert not cp.is_calibrated()

    def test_prediction_set(self):
        cp = ConformalPredictor(coverage=0.90)
        # All correct at high confidence
        data = [(0.95, True)] * 50 + [(0.3, False)] * 50
        cp.calibrate(data)

        outputs = ["A", "B", "C"]
        scores = [0.95, 0.50, 0.10]
        ps = cp.prediction_set(outputs, scores)
        assert isinstance(ps, PredictionSet)
        assert "A" in ps.items  # High confidence should be included
        assert ps.coverage_target == 0.90

    def test_prediction_set_not_calibrated(self):
        cp = ConformalPredictor()
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.prediction_set(["A"], [0.9])

    def test_mismatched_lengths(self):
        cp = ConformalPredictor()
        cp.calibrate([(0.9, True)] * 10)
        with pytest.raises(ValueError, match="same length"):
            cp.prediction_set(["A", "B"], [0.9])

    def test_invalid_coverage(self):
        with pytest.raises(ValueError, match="Coverage"):
            ConformalPredictor(coverage=1.5)

    def test_required_calibration_size(self):
        n = ConformalPredictor.required_calibration_size(0.90, 0.05)
        assert n > 0
        assert isinstance(n, int)

    def test_empty_calibration(self):
        cp = ConformalPredictor()
        with pytest.raises(ValueError, match="not be empty"):
            cp.calibrate([])


# ── Phase 14: Symbolic Inference ──────────────────────────────────


class TestTransitiveTemplate:
    def test_transitive_inference(self):
        template = TransitiveTemplate()
        claims = [
            {
                "claim_id": "c1",
                "subject_entity_id": "A",
                "predicate": "leads_to",
                "object_value": "B",
                "confidence": 0.9,
            },
            {
                "claim_id": "c2",
                "subject_entity_id": "B",
                "predicate": "leads_to",
                "object_value": "C",
                "confidence": 0.8,
            },
        ]
        inferred = template.match(claims)
        assert len(inferred) == 1
        assert inferred[0].subject == "A"
        assert inferred[0].object_ == "C"
        assert abs(inferred[0].confidence - 0.72) < 0.01
        assert inferred[0].inference_type == "transitive"

    def test_no_transitive_different_predicates(self):
        template = TransitiveTemplate()
        claims = [
            {
                "claim_id": "c1",
                "subject_entity_id": "A",
                "predicate": "leads_to",
                "object_value": "B",
                "confidence": 0.9,
            },
            {
                "claim_id": "c2",
                "subject_entity_id": "B",
                "predicate": "different",
                "object_value": "C",
                "confidence": 0.8,
            },
        ]
        inferred = template.match(claims)
        assert len(inferred) == 0


class TestAggregateTemplate:
    def test_aggregate_inference(self):
        template = AggregateTemplate()
        claims = [
            {
                "claim_id": "c1",
                "subject_entity_id": "X",
                "predicate": "color",
                "object_value": "blue",
                "confidence": 0.8,
            },
            {
                "claim_id": "c2",
                "subject_entity_id": "X",
                "predicate": "color",
                "object_value": "blue",
                "confidence": 0.9,
            },
        ]
        inferred = template.match(claims)
        assert len(inferred) == 1
        assert inferred[0].object_ == "blue"
        assert inferred[0].inference_type == "aggregate"
        assert inferred[0].confidence > 0.8

    def test_no_aggregate_single_claim(self):
        template = AggregateTemplate()
        claims = [
            {
                "claim_id": "c1",
                "subject_entity_id": "X",
                "predicate": "color",
                "object_value": "blue",
                "confidence": 0.8,
            },
        ]
        inferred = template.match(claims)
        assert len(inferred) == 0


class TestTemporalTemplate:
    def test_temporal_supersession(self):
        template = TemporalTemplate()
        claims = [
            {
                "claim_id": "c1",
                "subject_entity_id": "X",
                "predicate": "status",
                "object_value": "active",
                "confidence": 0.8,
                "created_at": "2024-01-01T00:00:00",
            },
            {
                "claim_id": "c2",
                "subject_entity_id": "X",
                "predicate": "status",
                "object_value": "inactive",
                "confidence": 0.9,
                "created_at": "2024-06-01T00:00:00",
            },
        ]
        inferred = template.match(claims)
        assert len(inferred) == 1
        assert inferred[0].object_ == "inactive"
        assert inferred[0].inference_type == "temporal"


class TestSymbolicInferenceEngine:
    @pytest.mark.asyncio
    async def test_infer_empty(self):
        engine = SymbolicInferenceEngine()
        results = await engine.infer([])
        assert results == []

    @pytest.mark.asyncio
    async def test_detect_inconsistencies(self):
        engine = SymbolicInferenceEngine()
        claims = [
            {
                "claim_id": "c1",
                "subject_entity_id": "X",
                "predicate": "color",
                "object_value": "blue",
                "confidence": 0.9,
            },
            {
                "claim_id": "c2",
                "subject_entity_id": "X",
                "predicate": "color",
                "object_value": "red",
                "confidence": 0.8,
            },
        ]
        inconsistencies = await engine.detect_inconsistencies(claims)
        assert len(inconsistencies) == 1
        assert len(inconsistencies[0].claim_ids) == 2

    @pytest.mark.asyncio
    async def test_no_inconsistencies(self):
        engine = SymbolicInferenceEngine()
        claims = [
            {
                "claim_id": "c1",
                "subject_entity_id": "X",
                "predicate": "color",
                "object_value": "blue",
                "confidence": 0.9,
            },
        ]
        inconsistencies = await engine.detect_inconsistencies(claims)
        assert len(inconsistencies) == 0

    def test_inference_chain(self):
        engine = SymbolicInferenceEngine()
        claims = [
            {"claim_id": "c1", "source_envelope_ids": []},
            {"claim_id": "c2", "source_envelope_ids": ["c1"]},
            {"claim_id": "c3", "source_envelope_ids": ["c2"]},
        ]
        chain = engine.get_inference_chain("c3", claims)
        assert len(chain) == 3
        # Chain should end with c3 (BFS reversed)
        assert chain[-1]["claim_id"] == "c3"


# ── Phase 14: Self-Correction ─────────────────────────────────────


class TestSelfCorrection:
    @pytest.mark.asyncio
    async def test_challenge_supersedes(self):
        engine = SelfCorrectionEngine()
        claim = {"claim_id": "c1", "confidence": 0.5}
        challenge = {"claim_id": "c2", "confidence": 0.8}
        result = await engine.evaluate_challenge(claim, challenge)
        assert result.action == "superseded"
        assert result.new_confidence == 0.8

    @pytest.mark.asyncio
    async def test_challenge_reinforced(self):
        engine = SelfCorrectionEngine()
        claim = {
            "claim_id": "c1",
            "confidence": 0.8,
            "source_envelope_ids": ["e1", "e2", "e3"],
        }
        challenge = {"claim_id": "c2", "confidence": 0.7}
        result = await engine.evaluate_challenge(claim, challenge)
        assert result.action == "reinforced"

    @pytest.mark.asyncio
    async def test_challenge_needs_investigation(self):
        engine = SelfCorrectionEngine()
        claim = {"claim_id": "c1", "confidence": 0.7}
        challenge = {"claim_id": "c2", "confidence": 0.75}
        result = await engine.evaluate_challenge(claim, challenge)
        assert result.action == "needs_investigation"

    @pytest.mark.asyncio
    async def test_resolve_contradiction(self):
        engine = SelfCorrectionEngine()
        claim_a = {"claim_id": "a", "confidence": 0.9}
        claim_b = {"claim_id": "b", "confidence": 0.5}
        result = await engine.resolve_contradiction(claim_a, claim_b)
        assert result.action == "superseded"
        assert result.original_claim_id == "b"

    def test_correction_stats(self):
        engine = SelfCorrectionEngine()
        stats = engine.get_correction_stats()
        assert stats["reinforced"] == 0
        assert stats["superseded"] == 0
        assert stats["needs_investigation"] == 0


# ── Phase 14: User Manager ────────────────────────────────────────


class TestUserManager:
    @pytest.mark.asyncio
    async def test_create_and_get(self):
        mgr = UserManager()
        user = await mgr.create_user("u1", "Alice")
        assert user.user_id == "u1"
        assert user.display_name == "Alice"

        fetched = await mgr.get_user("u1")
        assert fetched is not None
        assert fetched.user_id == "u1"

    @pytest.mark.asyncio
    async def test_create_duplicate(self):
        mgr = UserManager()
        await mgr.create_user("u1")
        with pytest.raises(ValueError, match="already exists"):
            await mgr.create_user("u1")

    @pytest.mark.asyncio
    async def test_list_users(self):
        mgr = UserManager()
        await mgr.create_user("u1")
        await mgr.create_user("u2")
        users = await mgr.list_users()
        assert len(users) == 2

    @pytest.mark.asyncio
    async def test_update_preferences(self):
        mgr = UserManager()
        await mgr.create_user("u1")
        await mgr.update_preferences("u1", {"theme": "dark"})
        user = await mgr.get_user("u1")
        assert user.preferences["theme"] == "dark"

    @pytest.mark.asyncio
    async def test_delete_user(self):
        mgr = UserManager()
        await mgr.create_user("u1")
        deleted = await mgr.delete_user("u1")
        assert deleted is True
        assert await mgr.get_user("u1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        mgr = UserManager()
        deleted = await mgr.delete_user("no_such")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_set_active_project(self):
        mgr = UserManager()
        await mgr.create_user("u1")
        await mgr.set_active_project("u1", "proj_1")
        user = await mgr.get_user("u1")
        assert user.active_project_id == "proj_1"

    @pytest.mark.asyncio
    async def test_get_user_context_missing(self):
        mgr = UserManager()
        with pytest.raises(ValueError, match="not found"):
            await mgr.get_user_context("missing")

    @pytest.mark.asyncio
    async def test_persistence(self):
        tmp = tempfile.mkdtemp()
        db_path = str(Path(tmp) / "users.db")
        mgr = UserManager(db_path=db_path)
        await mgr.create_user("u1", "Alice")

        # Reopen
        mgr2 = UserManager(db_path=db_path)
        user = await mgr2.get_user("u1")
        assert user is not None
        assert user.display_name == "Alice"


# ── Phase 15: Document Ingestion ──────────────────────────────────


class TestDocumentIngestor:
    def test_supported_formats(self):
        formats = DocumentIngestor.supported_formats()
        assert ".txt" in formats
        assert ".csv" in formats
        assert ".md" in formats
        assert ".html" in formats

    @pytest.mark.asyncio
    async def test_ingest_text(self):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w"
        )
        tmp.write("Hello world. This is a test document.")
        tmp.close()

        ingestor = DocumentIngestor()
        chunks = await ingestor.ingest(tmp.name)
        assert len(chunks) >= 1
        assert isinstance(chunks[0], DocumentChunk)
        assert "Hello world" in chunks[0].text

    @pytest.mark.asyncio
    async def test_ingest_csv(self):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        )
        tmp.write("name,age\nAlice,30\nBob,25\n")
        tmp.close()

        ingestor = DocumentIngestor()
        chunks = await ingestor.ingest(tmp.name)
        assert len(chunks) == 2
        assert "Alice" in chunks[0].text

    @pytest.mark.asyncio
    async def test_ingest_markdown(self):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".md", delete=False, mode="w"
        )
        tmp.write("# Section 1\nContent one.\n# Section 2\nContent two.\n")
        tmp.close()

        ingestor = DocumentIngestor()
        chunks = await ingestor.ingest(tmp.name)
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_ingest_html(self):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w"
        )
        tmp.write(
            "<html><body><p>Hello</p><script>alert(1)</script></body></html>"
        )
        tmp.close()

        ingestor = DocumentIngestor()
        chunks = await ingestor.ingest(tmp.name)
        assert len(chunks) >= 1
        assert "alert" not in chunks[0].text

    @pytest.mark.asyncio
    async def test_ingest_unsupported(self):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".xyz", delete=False, mode="w"
        )
        tmp.write("data")
        tmp.close()

        ingestor = DocumentIngestor()
        with pytest.raises(ValueError, match="Unsupported"):
            await ingestor.ingest(tmp.name)

    @pytest.mark.asyncio
    async def test_ingest_missing_file(self):
        ingestor = DocumentIngestor()
        with pytest.raises(FileNotFoundError):
            await ingestor.ingest("/nonexistent/file.txt")


# ── Phase 15: Voice Ingestor Models ──────────────────────────────


class TestVoiceModels:
    def test_voice_observation(self):
        obs = VoiceObservation(
            text="Hello world",
            speaker="speaker_0",
            timestamp_start=0.0,
            timestamp_end=1.5,
            confidence=0.95,
            source_file="test.wav",
        )
        assert obs.text == "Hello world"
        assert obs.speaker == "speaker_0"

    def test_transcript_segment(self):
        seg = TranscriptSegment(
            text="Test", start=0.0, end=1.0, confidence=0.9
        )
        assert seg.text == "Test"
        assert seg.speaker == ""

    def test_voice_ingestor_init(self):
        vi = VoiceIngestor(model_size="tiny")
        assert vi._model_size == "tiny"


# ── Phase 15: OCR Models ─────────────────────────────────────────


class TestOCRModels:
    def test_ocr_result(self):
        result = OCRResult(
            text="Hello",
            confidence=0.95,
            language="eng",
            source_file="test.png",
        )
        assert result.text == "Hello"
        assert result.confidence == 0.95

    def test_ocr_processor_init(self):
        proc = OCRProcessor(language="deu")
        assert proc._language == "deu"


# ── Phase 16: Channel Base ────────────────────────────────────────


class _TestAdapter(ChannelAdapter):
    """Concrete adapter for testing the abstract base."""

    async def start(self):
        self._running = True

    async def stop(self):
        self._running = False

    async def send(self, user_id, message, attachments=None):
        self._last_sent = (user_id, message)

    def _extract_text(self, raw_message):
        return raw_message.get("text", "")

    def _get_user_id(self, raw_message):
        return raw_message.get("user", "")


class TestChannelAdapter:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        adapter = _TestAdapter("test")
        await adapter.start()
        assert adapter.is_running
        await adapter.stop()
        assert not adapter.is_running

    @pytest.mark.asyncio
    async def test_receive(self):
        adapter = _TestAdapter("test")
        result = await adapter.receive(
            {"text": "Hello", "user": "u1"}
        )
        assert result is not None
        assert result["text"] == "Hello"
        assert result["user_id"] == "u1"
        assert result["channel"] == "test"

    @pytest.mark.asyncio
    async def test_receive_sanitized(self):
        sanitizer = MagicMock()
        sanitizer.sanitize.return_value = MagicMock(
            text="clean", risk_score=0.1
        )
        adapter = _TestAdapter("test", sanitizer=sanitizer)
        result = await adapter.receive({"text": "dirty", "user": "u1"})
        assert result["sanitized_text"] == "clean"

    @pytest.mark.asyncio
    async def test_receive_rejected(self):
        sanitizer = MagicMock()
        sanitizer.sanitize.return_value = MagicMock(
            text="bad", risk_score=0.9
        )
        adapter = _TestAdapter("test", sanitizer=sanitizer)
        result = await adapter.receive({"text": "inject", "user": "u1"})
        assert result is None

    def test_is_goal(self):
        adapter = _TestAdapter("test")
        assert adapter._is_goal("Goal: research X") is True
        assert adapter._is_goal("Research topic Y") is True
        assert adapter._is_goal("Hello world") is False


# ── Phase 16: Webhook Adapter ─────────────────────────────────────


class TestWebhookAdapter:
    def test_verify_signature_valid(self):
        adapter = WebhookAdapter(secret="mysecret")
        import hashlib
        import hmac as _hmac

        payload = b'{"test":"data"}'
        expected = _hmac.new(
            b"mysecret", payload, hashlib.sha256
        ).hexdigest()
        assert adapter.verify_signature(payload, expected)

    def test_verify_signature_with_prefix(self):
        adapter = WebhookAdapter(secret="mysecret")
        import hashlib
        import hmac as _hmac

        payload = b'{"test":"data"}'
        expected = _hmac.new(
            b"mysecret", payload, hashlib.sha256
        ).hexdigest()
        assert adapter.verify_signature(payload, f"sha256={expected}")

    def test_verify_signature_invalid(self):
        adapter = WebhookAdapter(secret="mysecret")
        assert not adapter.verify_signature(b"data", "invalid")

    def test_verify_no_secret(self):
        adapter = WebhookAdapter(secret="")
        assert adapter.verify_signature(b"any", "any")

    def test_extract_text(self):
        adapter = WebhookAdapter()
        assert adapter._extract_text({"text": "hello"}) == "hello"
        assert adapter._extract_text({"message": "hi"}) == "hi"
        assert adapter._extract_text({"body": "test"}) == "test"

    def test_extract_text_nested(self):
        adapter = WebhookAdapter()
        result = adapter._extract_text(
            {"message": {"text": "nested"}}
        )
        assert result == "nested"

    def test_get_user_id(self):
        adapter = WebhookAdapter()
        assert adapter._get_user_id({"user_id": "u1"}) == "u1"
        assert adapter._get_user_id({"sender": "bob"}) == "bob"
        assert adapter._get_user_id({}) == ""

    @pytest.mark.asyncio
    async def test_process_webhook(self):
        adapter = WebhookAdapter()
        result = await adapter.process_webhook(
            {"text": "hello", "user_id": "u1"}, {}
        )
        assert result is not None
        assert result["text"] == "hello"


# ── Phase 16: Notification Router ─────────────────────────────────


class TestNotificationRouter:
    @pytest.mark.asyncio
    async def test_register_and_notify(self):
        router = NotificationRouter()
        adapter = _TestAdapter("test_ch")
        adapter.send = AsyncMock()
        router.register_channel("test_ch", adapter)

        notified = await router.notify("u1", "alert", "Hello")
        assert "test_ch" in notified
        adapter.send.assert_called_once_with("u1", "Hello")

    @pytest.mark.asyncio
    async def test_notify_with_preferences(self):
        router = NotificationRouter()
        ch1 = _TestAdapter("ch1")
        ch1.send = AsyncMock()
        ch2 = _TestAdapter("ch2")
        ch2.send = AsyncMock()
        router.register_channel("ch1", ch1)
        router.register_channel("ch2", ch2)

        prefs = NotificationPreferences(
            user_id="u1",
            channels={"alert": ["ch1"]},
        )
        router.set_preferences("u1", prefs)

        notified = await router.notify("u1", "alert", "Hello")
        assert notified == ["ch1"]
        ch1.send.assert_called_once()
        ch2.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_deliver_result(self):
        router = NotificationRouter()
        adapter = _TestAdapter("ch")
        adapter.send = AsyncMock()
        router.register_channel("ch", adapter)

        notified = await router.deliver_result(
            "u1", "goal_123", {"status": "completed", "summary": "Done"}
        )
        assert "ch" in notified
        call_args = adapter.send.call_args[0]
        assert "goal_123" in call_args[1]

    def test_get_registered_channels(self):
        router = NotificationRouter()
        adapter = _TestAdapter("ch1")
        router.register_channel("ch1", adapter)
        assert router.get_registered_channels() == ["ch1"]

    def test_notification_priority_values(self):
        assert NotificationPriority.LOW.value == "low"
        assert NotificationPriority.CRITICAL.value == "critical"


# ── Phase 16: Browser Tool ────────────────────────────────────────


class TestBrowserTool:
    def test_browser_spec_exists(self):
        from qe.tools.browser import browser_spec

        assert browser_spec.name == "browser_navigate"
        assert browser_spec.requires_capability == "browser_control"


# ── Bus Protocol Updates ──────────────────────────────────────────


class TestBusProtocolUpdates:
    def test_new_topics_present(self):
        from qe.bus.protocol import TOPICS

        assert "voice.ingested" in TOPICS
        assert "voice.transcribed" in TOPICS
        assert "document.ingested" in TOPICS
        assert "document.parsed" in TOPICS
        assert "channel.message_received" in TOPICS
        assert "channel.message_sent" in TOPICS
        assert "notification.queued" in TOPICS
        assert "notification.delivered" in TOPICS
        assert "inference.claim_inferred" in TOPICS
        assert "inference.inconsistency_detected" in TOPICS
        assert "predictions.resolved" in TOPICS
        assert "monitor.scheduled" in TOPICS
        assert "system.service_restarted" in TOPICS
