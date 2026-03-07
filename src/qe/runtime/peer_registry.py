"""A2A Peer Registry — tracks known agent peers for outbound delegation.

Stores peer agent metadata (URL, name, capabilities, health status) in-memory
with optional SQLite persistence. Provides discovery, health checking, and
capability-based peer selection.
"""
from __future__ import annotations

import ipaddress
import logging
import time
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class PeerAgent(BaseModel):
    """A registered peer agent."""
    peer_id: str
    url: str
    name: str = ""
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    last_seen: float = Field(default_factory=time.time)
    healthy: bool = True
    version: str = ""
    permissions: str = "restricted"  # preset: restricted / standard / autonomous


class PeerRegistry:
    """In-memory peer agent registry with health tracking."""

    def __init__(self) -> None:
        self._peers: dict[str, PeerAgent] = {}

    def register(
        self,
        url: str,
        name: str = "",
        description: str = "",
        peer_id: str | None = None,
    ) -> PeerAgent:
        """Register a peer agent. Uses URL hash as peer_id if not provided."""
        import hashlib
        pid = peer_id or hashlib.sha256(url.encode()).hexdigest()[:12]
        peer = PeerAgent(
            peer_id=pid,
            url=url.rstrip("/"),
            name=name,
            description=description,
        )
        self._peers[pid] = peer
        log.info("peer_registry.registered peer_id=%s url=%s", pid, url)
        return peer

    def unregister(self, peer_id: str) -> bool:
        """Remove a peer by ID. Returns True if found."""
        if peer_id in self._peers:
            del self._peers[peer_id]
            log.info("peer_registry.unregistered peer_id=%s", peer_id)
            return True
        return False

    def get(self, peer_id: str) -> PeerAgent | None:
        """Get a peer by ID."""
        return self._peers.get(peer_id)

    def list_peers(self) -> list[PeerAgent]:
        """List all registered peers."""
        return list(self._peers.values())

    def list_healthy(self) -> list[PeerAgent]:
        """List only healthy peers."""
        return [p for p in self._peers.values() if p.healthy]

    def find_by_capability(self, capability: str) -> list[PeerAgent]:
        """Find peers that advertise a specific capability."""
        return [
            p for p in self._peers.values()
            if p.healthy and capability in p.capabilities
        ]

    def find_by_skill(self, skill: str) -> list[PeerAgent]:
        """Find peers that advertise a specific skill."""
        return [
            p for p in self._peers.values()
            if p.healthy and skill in p.skills
        ]

    def mark_healthy(self, peer_id: str) -> None:
        """Mark a peer as healthy."""
        if peer := self._peers.get(peer_id):
            peer.healthy = True
            peer.last_seen = time.time()

    def mark_unhealthy(self, peer_id: str) -> None:
        """Mark a peer as unhealthy."""
        if peer := self._peers.get(peer_id):
            peer.healthy = False

    def set_peer_permissions(self, peer_id: str, preset: str) -> bool:
        """Set permission preset for a peer. Returns True if peer found."""
        if preset not in ("restricted", "standard", "autonomous"):
            preset = "restricted"
        if peer := self._peers.get(peer_id):
            peer.permissions = preset
            log.info("peer_registry.permissions_set peer_id=%s preset=%s", peer_id, preset)
            return True
        return False

    @staticmethod
    def _validate_peer_url(url: str) -> str:
        """Validate a peer URL to prevent SSRF attacks.

        Raises ValueError if the URL is unsafe (private IP, bad scheme, etc.).
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme!r} (only http/https allowed)")
        hostname = parsed.hostname or ""
        if not hostname:
            raise ValueError("URL missing hostname")
        # Block obviously dangerous hostnames
        _blocked = {"localhost", "127.0.0.1", "0.0.0.0", "[::1]", "metadata.google.internal"}
        if hostname.lower() in _blocked:
            raise ValueError(f"Blocked hostname: {hostname}")
        # Block private/reserved IP ranges
        try:
            addr = ipaddress.ip_address(hostname)
            if addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local:
                raise ValueError(f"Blocked private/reserved IP: {addr}")
        except ValueError as exc:
            if "Blocked" in str(exc):
                raise
            # Not an IP literal — hostname is fine
        return url

    async def discover_and_register(self, url: str) -> PeerAgent | None:
        """Discover a remote agent and register it.

        Fetches /.well-known/agent.json from the URL, extracts metadata,
        and registers the peer.
        """
        self._validate_peer_url(url)

        from qe.runtime.a2a_client import A2AClient

        client = A2AClient(url, timeout=10.0)
        try:
            card = await client.discover()
            peer = self.register(
                url=url,
                name=card.get("name", ""),
                description=card.get("description", ""),
            )
            # Extract capabilities and skills from agent card
            caps = card.get("capabilities", {})
            if isinstance(caps, dict):
                peer.capabilities = caps.get("intents", [])
            elif isinstance(caps, list):
                peer.capabilities = caps
            skills = card.get("skills", [])
            if isinstance(skills, list):
                peer.skills = [
                    s.get("name", s) if isinstance(s, dict) else str(s)
                    for s in skills
                ]
            peer.version = card.get("version", "")
            peer.healthy = True
            peer.last_seen = time.time()
            return peer
        except Exception as exc:
            log.warning(
                "peer_registry.discover_failed url=%s error=%s",
                url, exc,
            )
            return None

    async def check_health(self, peer_id: str) -> bool:
        """Check if a peer is reachable. Updates healthy status."""
        peer = self._peers.get(peer_id)
        if not peer:
            return False

        from qe.runtime.a2a_client import A2AClient
        client = A2AClient(peer.url, timeout=5.0)
        try:
            await client.discover()
            self.mark_healthy(peer_id)
            return True
        except Exception:
            self.mark_unhealthy(peer_id)
            return False

    def status(self) -> dict[str, Any]:
        """Return registry status summary."""
        peers = self.list_peers()
        return {
            "total_peers": len(peers),
            "healthy_peers": sum(1 for p in peers if p.healthy),
            "unhealthy_peers": sum(1 for p in peers if not p.healthy),
            "peers": [p.model_dump() for p in peers],
        }
