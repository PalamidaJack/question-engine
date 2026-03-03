"""Tests for PeerRegistry — PeerAgent model, register/unregister, capability/skill
lookup, health tracking, discovery via mocked A2AClient, and status summary."""

from __future__ import annotations

import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.runtime.peer_registry import PeerAgent, PeerRegistry

# ═══════════════════════════════════════════════════════════════════════════
# TestPeerAgentModel
# ═══════════════════════════════════════════════════════════════════════════


class TestPeerAgentModel:
    """PeerAgent Pydantic model: construction, defaults, serialization."""

    def test_minimal_construction(self):
        """peer_id and url are the only required fields."""
        agent = PeerAgent(peer_id="abc123", url="http://peer.local")
        assert agent.peer_id == "abc123"
        assert agent.url == "http://peer.local"

    def test_defaults(self):
        """Optional fields have sensible defaults."""
        agent = PeerAgent(peer_id="p1", url="http://x")
        assert agent.name == ""
        assert agent.description == ""
        assert agent.capabilities == []
        assert agent.skills == []
        assert agent.healthy is True
        assert agent.version == ""
        # last_seen should be recent (within 5 seconds)
        assert abs(agent.last_seen - time.time()) < 5.0

    def test_full_construction(self):
        """All fields can be set explicitly."""
        agent = PeerAgent(
            peer_id="peer-42",
            url="http://agent.io",
            name="Agent 42",
            description="A test agent",
            capabilities=["research", "code"],
            skills=["web_search", "summarize"],
            last_seen=1000.0,
            healthy=False,
            version="1.2.3",
        )
        assert agent.peer_id == "peer-42"
        assert agent.name == "Agent 42"
        assert agent.description == "A test agent"
        assert agent.capabilities == ["research", "code"]
        assert agent.skills == ["web_search", "summarize"]
        assert agent.last_seen == 1000.0
        assert agent.healthy is False
        assert agent.version == "1.2.3"

    def test_model_dump(self):
        """Serialization via model_dump includes all fields."""
        agent = PeerAgent(peer_id="p1", url="http://x", name="Test")
        data = agent.model_dump()
        assert data["peer_id"] == "p1"
        assert data["url"] == "http://x"
        assert data["name"] == "Test"
        assert "capabilities" in data
        assert "skills" in data
        assert "healthy" in data
        assert "last_seen" in data
        assert "version" in data

    def test_model_dump_roundtrip(self):
        """model_dump -> PeerAgent reconstruction preserves data."""
        original = PeerAgent(
            peer_id="rt",
            url="http://roundtrip",
            name="RT",
            capabilities=["cap1"],
            skills=["sk1"],
            version="0.1",
        )
        rebuilt = PeerAgent(**original.model_dump())
        assert rebuilt.peer_id == original.peer_id
        assert rebuilt.url == original.url
        assert rebuilt.capabilities == original.capabilities
        assert rebuilt.skills == original.skills

    def test_mutable_fields(self):
        """healthy and last_seen can be mutated after construction."""
        agent = PeerAgent(peer_id="m1", url="http://m")
        assert agent.healthy is True
        agent.healthy = False
        assert agent.healthy is False
        agent.last_seen = 999.0
        assert agent.last_seen == 999.0


# ═══════════════════════════════════════════════════════════════════════════
# TestPeerRegistryRegisterUnregister
# ═══════════════════════════════════════════════════════════════════════════


class TestPeerRegistryRegisterUnregister:
    """register, unregister, get, list_peers."""

    def test_register_returns_peer(self):
        """register() returns a PeerAgent with given metadata."""
        reg = PeerRegistry()
        peer = reg.register(url="http://agent.local", name="Agent A")
        assert isinstance(peer, PeerAgent)
        assert peer.url == "http://agent.local"
        assert peer.name == "Agent A"

    def test_register_strips_trailing_slash(self):
        """URL trailing slash is stripped on registration."""
        reg = PeerRegistry()
        peer = reg.register(url="http://agent.local/")
        assert peer.url == "http://agent.local"

    def test_register_auto_generates_peer_id(self):
        """When no peer_id is provided, a SHA-256 hash of the URL is used."""
        reg = PeerRegistry()
        url = "http://agent.local"
        peer = reg.register(url=url)
        expected_id = hashlib.sha256(url.encode()).hexdigest()[:12]
        assert peer.peer_id == expected_id

    def test_register_explicit_peer_id(self):
        """Explicit peer_id overrides the auto-generated hash."""
        reg = PeerRegistry()
        peer = reg.register(url="http://x", peer_id="custom-id")
        assert peer.peer_id == "custom-id"

    def test_register_overwrites_same_id(self):
        """Registering with the same peer_id replaces the previous entry."""
        reg = PeerRegistry()
        reg.register(url="http://old", peer_id="dup", name="Old")
        peer = reg.register(url="http://new", peer_id="dup", name="New")
        assert peer.name == "New"
        assert peer.url == "http://new"
        assert len(reg.list_peers()) == 1

    def test_get_existing(self):
        """get() returns the peer for a known peer_id."""
        reg = PeerRegistry()
        peer = reg.register(url="http://x", peer_id="p1")
        assert reg.get("p1") is peer

    def test_get_missing_returns_none(self):
        """get() returns None for an unknown peer_id."""
        reg = PeerRegistry()
        assert reg.get("nonexistent") is None

    def test_unregister_existing(self):
        """unregister() removes the peer and returns True."""
        reg = PeerRegistry()
        reg.register(url="http://x", peer_id="p1")
        assert reg.unregister("p1") is True
        assert reg.get("p1") is None

    def test_unregister_missing(self):
        """unregister() returns False for an unknown peer_id."""
        reg = PeerRegistry()
        assert reg.unregister("ghost") is False

    def test_list_peers_empty(self):
        """list_peers() returns empty list initially."""
        reg = PeerRegistry()
        assert reg.list_peers() == []

    def test_list_peers_multiple(self):
        """list_peers() returns all registered peers."""
        reg = PeerRegistry()
        reg.register(url="http://a", peer_id="a")
        reg.register(url="http://b", peer_id="b")
        reg.register(url="http://c", peer_id="c")
        peers = reg.list_peers()
        assert len(peers) == 3
        ids = {p.peer_id for p in peers}
        assert ids == {"a", "b", "c"}


# ═══════════════════════════════════════════════════════════════════════════
# TestPeerRegistryHealthyFilter
# ═══════════════════════════════════════════════════════════════════════════


class TestPeerRegistryHealthyFilter:
    """list_healthy, mark_healthy, mark_unhealthy."""

    def test_list_healthy_all_healthy(self):
        """All peers are healthy by default after registration."""
        reg = PeerRegistry()
        reg.register(url="http://a", peer_id="a")
        reg.register(url="http://b", peer_id="b")
        assert len(reg.list_healthy()) == 2

    def test_list_healthy_excludes_unhealthy(self):
        """list_healthy() excludes peers marked unhealthy."""
        reg = PeerRegistry()
        reg.register(url="http://a", peer_id="a")
        reg.register(url="http://b", peer_id="b")
        reg.mark_unhealthy("b")
        healthy = reg.list_healthy()
        assert len(healthy) == 1
        assert healthy[0].peer_id == "a"

    def test_mark_healthy_updates_last_seen(self):
        """mark_healthy() sets healthy=True and updates last_seen."""
        reg = PeerRegistry()
        peer = reg.register(url="http://a", peer_id="a")
        peer.healthy = False
        peer.last_seen = 0.0
        reg.mark_healthy("a")
        assert peer.healthy is True
        assert peer.last_seen > 0.0

    def test_mark_unhealthy_sets_flag(self):
        """mark_unhealthy() sets healthy=False."""
        reg = PeerRegistry()
        peer = reg.register(url="http://a", peer_id="a")
        assert peer.healthy is True
        reg.mark_unhealthy("a")
        assert peer.healthy is False

    def test_mark_healthy_unknown_peer_is_noop(self):
        """mark_healthy() on unknown peer_id does nothing (no error)."""
        reg = PeerRegistry()
        reg.mark_healthy("ghost")  # should not raise

    def test_mark_unhealthy_unknown_peer_is_noop(self):
        """mark_unhealthy() on unknown peer_id does nothing (no error)."""
        reg = PeerRegistry()
        reg.mark_unhealthy("ghost")  # should not raise

    def test_mark_healthy_then_unhealthy_then_healthy(self):
        """Health status can be toggled multiple times."""
        reg = PeerRegistry()
        peer = reg.register(url="http://a", peer_id="a")
        assert peer.healthy is True
        reg.mark_unhealthy("a")
        assert peer.healthy is False
        reg.mark_healthy("a")
        assert peer.healthy is True


# ═══════════════════════════════════════════════════════════════════════════
# TestFindByCapability
# ═══════════════════════════════════════════════════════════════════════════


class TestFindByCapability:
    """find_by_capability — filters by capability and healthy status."""

    def test_find_matching_capability(self):
        """Returns peers that have the requested capability."""
        reg = PeerRegistry()
        p1 = reg.register(url="http://a", peer_id="a")
        p1.capabilities = ["research", "code"]
        p2 = reg.register(url="http://b", peer_id="b")
        p2.capabilities = ["research"]
        p3 = reg.register(url="http://c", peer_id="c")
        p3.capabilities = ["code"]

        found = reg.find_by_capability("research")
        ids = {p.peer_id for p in found}
        assert ids == {"a", "b"}

    def test_find_no_match(self):
        """Returns empty list when no peer has the capability."""
        reg = PeerRegistry()
        p1 = reg.register(url="http://a", peer_id="a")
        p1.capabilities = ["code"]
        assert reg.find_by_capability("unknown_cap") == []

    def test_find_excludes_unhealthy(self):
        """Unhealthy peers are excluded from capability search."""
        reg = PeerRegistry()
        p1 = reg.register(url="http://a", peer_id="a")
        p1.capabilities = ["research"]
        p2 = reg.register(url="http://b", peer_id="b")
        p2.capabilities = ["research"]
        reg.mark_unhealthy("b")

        found = reg.find_by_capability("research")
        assert len(found) == 1
        assert found[0].peer_id == "a"

    def test_find_empty_registry(self):
        """Returns empty list when registry has no peers."""
        reg = PeerRegistry()
        assert reg.find_by_capability("anything") == []


# ═══════════════════════════════════════════════════════════════════════════
# TestFindBySkill
# ═══════════════════════════════════════════════════════════════════════════


class TestFindBySkill:
    """find_by_skill — filters by skill name and healthy status."""

    def test_find_matching_skill(self):
        """Returns peers that have the requested skill."""
        reg = PeerRegistry()
        p1 = reg.register(url="http://a", peer_id="a")
        p1.skills = ["web_search", "summarize"]
        p2 = reg.register(url="http://b", peer_id="b")
        p2.skills = ["web_search"]
        p3 = reg.register(url="http://c", peer_id="c")
        p3.skills = ["code_execute"]

        found = reg.find_by_skill("web_search")
        ids = {p.peer_id for p in found}
        assert ids == {"a", "b"}

    def test_find_no_match(self):
        """Returns empty list when no peer has the skill."""
        reg = PeerRegistry()
        p1 = reg.register(url="http://a", peer_id="a")
        p1.skills = ["summarize"]
        assert reg.find_by_skill("unknown_skill") == []

    def test_find_excludes_unhealthy(self):
        """Unhealthy peers are excluded from skill search."""
        reg = PeerRegistry()
        p1 = reg.register(url="http://a", peer_id="a")
        p1.skills = ["web_search"]
        p2 = reg.register(url="http://b", peer_id="b")
        p2.skills = ["web_search"]
        reg.mark_unhealthy("a")

        found = reg.find_by_skill("web_search")
        assert len(found) == 1
        assert found[0].peer_id == "b"

    def test_find_empty_registry(self):
        """Returns empty list when registry has no peers."""
        reg = PeerRegistry()
        assert reg.find_by_skill("anything") == []


# ═══════════════════════════════════════════════════════════════════════════
# TestDiscoverAndRegister
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscoverAndRegister:
    """discover_and_register — mocks A2AClient.discover() to return agent card data."""

    @pytest.mark.asyncio
    async def test_discover_basic_card(self):
        """Discovers a remote agent and registers it with name/description."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={
            "name": "Remote Agent",
            "description": "A remote peer",
            "version": "2.0.0",
        })

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            peer = await reg.discover_and_register("http://remote.agent")

        assert peer is not None
        assert peer.name == "Remote Agent"
        assert peer.description == "A remote peer"
        assert peer.version == "2.0.0"
        assert peer.healthy is True
        assert peer.url == "http://remote.agent"

    @pytest.mark.asyncio
    async def test_discover_capabilities_as_dict(self):
        """Agent card with capabilities as dict extracts intents list."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={
            "name": "Cap Agent",
            "capabilities": {
                "intents": ["research", "delegate"],
                "streaming": True,
            },
        })

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            peer = await reg.discover_and_register("http://cap.agent")

        assert peer is not None
        assert peer.capabilities == ["research", "delegate"]

    @pytest.mark.asyncio
    async def test_discover_capabilities_as_list(self):
        """Agent card with capabilities as list uses it directly."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={
            "name": "List Cap",
            "capabilities": ["analyze", "summarize"],
        })

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            peer = await reg.discover_and_register("http://listcap.agent")

        assert peer is not None
        assert peer.capabilities == ["analyze", "summarize"]

    @pytest.mark.asyncio
    async def test_discover_skills_as_dicts(self):
        """Agent card with skills as list of dicts extracts name field."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={
            "name": "Skill Agent",
            "skills": [
                {"name": "web_search", "description": "Search the web"},
                {"name": "code_execute", "description": "Run code"},
            ],
        })

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            peer = await reg.discover_and_register("http://skill.agent")

        assert peer is not None
        assert peer.skills == ["web_search", "code_execute"]

    @pytest.mark.asyncio
    async def test_discover_skills_as_strings(self):
        """Agent card with skills as list of strings uses them directly."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={
            "name": "Str Skill",
            "skills": ["summarize", "translate"],
        })

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            peer = await reg.discover_and_register("http://strskill.agent")

        assert peer is not None
        assert peer.skills == ["summarize", "translate"]

    @pytest.mark.asyncio
    async def test_discover_failure_returns_none(self):
        """When discover() raises an exception, returns None without crashing."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            peer = await reg.discover_and_register("http://down.agent")

        assert peer is None
        assert len(reg.list_peers()) == 0

    @pytest.mark.asyncio
    async def test_discover_registers_in_peers(self):
        """A successfully discovered agent appears in list_peers()."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={
            "name": "Discovered",
        })

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            peer = await reg.discover_and_register("http://disc.agent")

        assert peer is not None
        assert len(reg.list_peers()) == 1
        assert reg.list_peers()[0].peer_id == peer.peer_id

    @pytest.mark.asyncio
    async def test_discover_empty_card(self):
        """An empty agent card still registers successfully with defaults."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={})

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            peer = await reg.discover_and_register("http://empty.agent")

        assert peer is not None
        assert peer.name == ""
        assert peer.description == ""
        assert peer.capabilities == []
        assert peer.skills == []
        assert peer.version == ""

    @pytest.mark.asyncio
    async def test_discover_uses_correct_timeout(self):
        """A2AClient is constructed with timeout=10.0 for discovery."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={"name": "T"})

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client) as mock_cls:
            reg = PeerRegistry()
            await reg.discover_and_register("http://t.agent")
            mock_cls.assert_called_once_with("http://t.agent", timeout=10.0)


# ═══════════════════════════════════════════════════════════════════════════
# TestCheckHealth
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckHealth:
    """check_health — mocks A2AClient.discover() for health probes."""

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        """Successful discover() marks peer as healthy and returns True."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={"name": "alive"})

        reg = PeerRegistry()
        peer = reg.register(url="http://alive.agent", peer_id="alive")
        peer.healthy = False  # start unhealthy

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            result = await reg.check_health("alive")

        assert result is True
        assert peer.healthy is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self):
        """Failed discover() marks peer as unhealthy and returns False."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(side_effect=ConnectionError("timeout"))

        reg = PeerRegistry()
        peer = reg.register(url="http://dead.agent", peer_id="dead")
        assert peer.healthy is True  # starts healthy

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            result = await reg.check_health("dead")

        assert result is False
        assert peer.healthy is False

    @pytest.mark.asyncio
    async def test_check_health_unknown_peer(self):
        """check_health() for unknown peer_id returns False without calling A2AClient."""
        reg = PeerRegistry()
        result = await reg.check_health("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_health_updates_last_seen_on_success(self):
        """Successful health check updates last_seen timestamp."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={})

        reg = PeerRegistry()
        peer = reg.register(url="http://ts.agent", peer_id="ts")
        peer.last_seen = 0.0

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            await reg.check_health("ts")

        assert peer.last_seen > 0.0

    @pytest.mark.asyncio
    async def test_check_health_uses_correct_timeout(self):
        """A2AClient is constructed with timeout=5.0 for health checks."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={})

        reg = PeerRegistry()
        reg.register(url="http://t.agent", peer_id="t")

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client) as mock_cls:
            await reg.check_health("t")
            mock_cls.assert_called_once_with("http://t.agent", timeout=5.0)

    @pytest.mark.asyncio
    async def test_check_health_uses_stored_url(self):
        """Health check uses the peer's stored URL (after trailing slash strip)."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={})

        reg = PeerRegistry()
        reg.register(url="http://stored.agent/", peer_id="s")

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client) as mock_cls:
            await reg.check_health("s")
            # URL should have trailing slash stripped at registration time
            mock_cls.assert_called_once_with("http://stored.agent", timeout=5.0)


# ═══════════════════════════════════════════════════════════════════════════
# TestStatus
# ═══════════════════════════════════════════════════════════════════════════


class TestStatus:
    """status() — summary dict with counts and peer details."""

    def test_status_empty(self):
        """Empty registry returns zero counts and empty peers list."""
        reg = PeerRegistry()
        s = reg.status()
        assert s["total_peers"] == 0
        assert s["healthy_peers"] == 0
        assert s["unhealthy_peers"] == 0
        assert s["peers"] == []

    def test_status_all_healthy(self):
        """All healthy peers: healthy_peers == total_peers, unhealthy == 0."""
        reg = PeerRegistry()
        reg.register(url="http://a", peer_id="a")
        reg.register(url="http://b", peer_id="b")
        s = reg.status()
        assert s["total_peers"] == 2
        assert s["healthy_peers"] == 2
        assert s["unhealthy_peers"] == 0

    def test_status_mixed_health(self):
        """Counts reflect actual healthy/unhealthy split."""
        reg = PeerRegistry()
        reg.register(url="http://a", peer_id="a")
        reg.register(url="http://b", peer_id="b")
        reg.register(url="http://c", peer_id="c")
        reg.mark_unhealthy("b")
        s = reg.status()
        assert s["total_peers"] == 3
        assert s["healthy_peers"] == 2
        assert s["unhealthy_peers"] == 1

    def test_status_peers_are_dicts(self):
        """Peers in status are serialized dicts (model_dump output)."""
        reg = PeerRegistry()
        reg.register(url="http://a", peer_id="a", name="Alpha")
        s = reg.status()
        assert len(s["peers"]) == 1
        peer_dict = s["peers"][0]
        assert isinstance(peer_dict, dict)
        assert peer_dict["peer_id"] == "a"
        assert peer_dict["name"] == "Alpha"
        assert peer_dict["url"] == "http://a"

    def test_status_all_unhealthy(self):
        """When all peers are unhealthy: healthy == 0, unhealthy == total."""
        reg = PeerRegistry()
        reg.register(url="http://a", peer_id="a")
        reg.register(url="http://b", peer_id="b")
        reg.mark_unhealthy("a")
        reg.mark_unhealthy("b")
        s = reg.status()
        assert s["total_peers"] == 2
        assert s["healthy_peers"] == 0
        assert s["unhealthy_peers"] == 2

    def test_status_after_unregister(self):
        """Unregistered peers do not appear in status."""
        reg = PeerRegistry()
        reg.register(url="http://a", peer_id="a")
        reg.register(url="http://b", peer_id="b")
        reg.unregister("a")
        s = reg.status()
        assert s["total_peers"] == 1
        assert len(s["peers"]) == 1
        assert s["peers"][0]["peer_id"] == "b"


# ═══════════════════════════════════════════════════════════════════════════
# TestIntegration
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end scenarios combining multiple operations."""

    @pytest.mark.asyncio
    async def test_discover_then_find_by_capability(self):
        """Discovered agent with capabilities is findable via find_by_capability."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={
            "name": "Research Bot",
            "capabilities": {"intents": ["research", "analyze"]},
            "skills": [{"name": "web_search"}],
        })

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            await reg.discover_and_register("http://research.bot")

        found = reg.find_by_capability("research")
        assert len(found) == 1
        assert found[0].name == "Research Bot"

    @pytest.mark.asyncio
    async def test_discover_then_find_by_skill(self):
        """Discovered agent with skills is findable via find_by_skill."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={
            "name": "Search Bot",
            "skills": ["web_search", "summarize"],
        })

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            await reg.discover_and_register("http://search.bot")

        found = reg.find_by_skill("web_search")
        assert len(found) == 1
        assert found[0].name == "Search Bot"

    @pytest.mark.asyncio
    async def test_discover_health_check_cycle(self):
        """Discover -> mark_unhealthy -> check_health recovers healthy state."""
        mock_client = MagicMock()
        mock_client.discover = AsyncMock(return_value={"name": "Cycle Agent"})

        with patch("qe.runtime.a2a_client.A2AClient", return_value=mock_client):
            reg = PeerRegistry()
            peer = await reg.discover_and_register("http://cycle.agent")
            assert peer is not None
            assert peer.healthy is True

            reg.mark_unhealthy(peer.peer_id)
            assert peer.healthy is False
            assert len(reg.list_healthy()) == 0

            result = await reg.check_health(peer.peer_id)
            assert result is True
            assert peer.healthy is True
            assert len(reg.list_healthy()) == 1

    def test_register_multiple_find_and_status(self):
        """Register several peers, query by capability, verify status counts."""
        reg = PeerRegistry()

        p1 = reg.register(url="http://a", peer_id="a", name="Alpha")
        p1.capabilities = ["research"]
        p1.skills = ["web_search"]

        p2 = reg.register(url="http://b", peer_id="b", name="Beta")
        p2.capabilities = ["code"]
        p2.skills = ["code_execute"]

        p3 = reg.register(url="http://c", peer_id="c", name="Gamma")
        p3.capabilities = ["research", "code"]
        p3.skills = ["web_search", "code_execute"]

        reg.mark_unhealthy("b")

        # Capability search
        researchers = reg.find_by_capability("research")
        assert len(researchers) == 2  # a and c (b is unhealthy)
        assert {p.peer_id for p in researchers} == {"a", "c"}

        coders = reg.find_by_capability("code")
        assert len(coders) == 1  # only c (b is unhealthy)
        assert coders[0].peer_id == "c"

        # Skill search
        searchers = reg.find_by_skill("web_search")
        assert len(searchers) == 2
        assert {p.peer_id for p in searchers} == {"a", "c"}

        # Status
        s = reg.status()
        assert s["total_peers"] == 3
        assert s["healthy_peers"] == 2
        assert s["unhealthy_peers"] == 1
