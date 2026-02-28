"""API key authentication and authorization for QE endpoints.

Provides FastAPI dependency injection for protecting routes with
API key validation and scope-based access control.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from enum import StrEnum

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

log = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Scope(StrEnum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


# Scope hierarchy: admin > write > read
_SCOPE_HIERARCHY: dict[Scope, set[Scope]] = {
    Scope.ADMIN: {Scope.ADMIN, Scope.WRITE, Scope.READ},
    Scope.WRITE: {Scope.WRITE, Scope.READ},
    Scope.READ: {Scope.READ},
}


class AuthContext:
    """Resolved authentication context for a request."""

    def __init__(self, scope: Scope, key_id: str = "") -> None:
        self.scope = scope
        self.key_id = key_id

    def has_scope(self, required: Scope) -> bool:
        return required in _SCOPE_HIERARCHY.get(self.scope, set())


class AuthProvider:
    """Manages API keys and validates requests.

    Keys are stored as SHA-256 hashes to avoid plaintext exposure in memory.
    """

    def __init__(self) -> None:
        self._keys: dict[str, tuple[Scope, str]] = {}  # hash -> (scope, key_id)
        self._require_auth = False

    def configure(
        self,
        *,
        api_key: str | None = None,
        admin_api_key: str | None = None,
        require_auth: bool = False,
    ) -> None:
        """Configure authentication from settings."""
        self._keys.clear()
        self._require_auth = require_auth

        if api_key:
            h = self._hash_key(api_key)
            self._keys[h] = (Scope.WRITE, "default")
        if admin_api_key:
            h = self._hash_key(admin_api_key)
            self._keys[h] = (Scope.ADMIN, "admin")

        if self._keys:
            self._require_auth = True

        log.debug(
            "auth.configured require_auth=%s keys=%d",
            self._require_auth,
            len(self._keys),
        )

    @property
    def enabled(self) -> bool:
        return self._require_auth

    def validate_key(self, key: str) -> AuthContext | None:
        """Validate an API key and return its auth context."""
        h = self._hash_key(key)
        entry = self._keys.get(h)
        if entry is None:
            return None
        scope, key_id = entry
        return AuthContext(scope=scope, key_id=key_id)

    @staticmethod
    def generate_key() -> str:
        """Generate a new random API key."""
        return f"qe_{secrets.token_urlsafe(32)}"

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()


# ── Singleton ──────────────────────────────────────────────────────────────

_auth_provider = AuthProvider()


def get_auth_provider() -> AuthProvider:
    return _auth_provider


# ── FastAPI Dependencies ───────────────────────────────────────────────────


async def _resolve_auth(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> AuthContext:
    """Resolve authentication for a request.

    When auth is disabled, returns a full-access context.
    When auth is enabled, validates the X-API-Key header.
    """
    provider = get_auth_provider()

    if not provider.enabled:
        return AuthContext(scope=Scope.ADMIN, key_id="no_auth")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={"code": "QE_SECURITY_AUTH_FAILED", "message": "X-API-Key header required"},
        )

    ctx = provider.validate_key(api_key)
    if ctx is None:
        log.warning(
            "auth.invalid_key ip=%s path=%s",
            request.client.host if request.client else "unknown",
            request.url.path,
        )
        raise HTTPException(
            status_code=401,
            detail={"code": "QE_SECURITY_AUTH_FAILED", "message": "Invalid API key"},
        )

    return ctx


def require_read(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:  # noqa: B008
    """Dependency: requires at least read scope."""
    if not auth.has_scope(Scope.READ):
        raise HTTPException(status_code=403, detail="Insufficient scope: read required")
    return auth


def require_write(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:  # noqa: B008
    """Dependency: requires at least write scope."""
    if not auth.has_scope(Scope.WRITE):
        raise HTTPException(status_code=403, detail="Insufficient scope: write required")
    return auth


def require_admin(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:  # noqa: B008
    """Dependency: requires admin scope."""
    if not auth.has_scope(Scope.ADMIN):
        raise HTTPException(status_code=403, detail="Insufficient scope: admin required")
    return auth
