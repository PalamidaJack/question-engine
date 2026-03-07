"""Knowledge Tree — hierarchical filesystem-like view of claims.

Organizes claims into a domain→topic→claims tree structure,
browsable via cd/ls/cat/tree-like operations.
Gated behind ``knowledge_filesystem`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A node in the knowledge tree."""

    name: str
    path: str
    node_type: str = "directory"  # directory | claim
    children: dict[str, TreeNode] = field(default_factory=dict)
    claim_data: dict[str, Any] | None = None

    @property
    def is_directory(self) -> bool:
        return self.node_type == "directory"

    @property
    def child_count(self) -> int:
        return len(self.children)


class KnowledgeTree:
    """Hierarchical tree view over knowledge claims."""

    def __init__(self) -> None:
        self._root = TreeNode(name="/", path="/")

    def build(self, claims: list[Any]) -> None:
        """Build the tree from a list of claims."""
        self._root = TreeNode(name="/", path="/")
        for claim in claims:
            entity = getattr(claim, "subject_entity_id", "misc")
            predicate = getattr(claim, "predicate", "unknown")
            claim_id = getattr(claim, "claim_id", "")
            conf = getattr(claim, "confidence", 0.5)
            obj_val = getattr(claim, "object_value", "")

            # Create entity directory
            entity_key = entity.lower().replace(" ", "_")
            if entity_key not in self._root.children:
                self._root.children[entity_key] = TreeNode(
                    name=entity_key,
                    path=f"/{entity_key}",
                )

            entity_node = self._root.children[entity_key]

            # Create predicate directory
            pred_key = predicate.lower().replace(" ", "_")
            if pred_key not in entity_node.children:
                entity_node.children[pred_key] = TreeNode(
                    name=pred_key,
                    path=f"/{entity_key}/{pred_key}",
                )

            pred_node = entity_node.children[pred_key]

            # Add claim leaf
            claim_key = claim_id or f"claim_{pred_node.child_count}"
            pred_node.children[claim_key] = TreeNode(
                name=claim_key,
                path=f"/{entity_key}/{pred_key}/{claim_key}",
                node_type="claim",
                claim_data={
                    "claim_id": claim_id,
                    "entity": entity,
                    "predicate": predicate,
                    "object_value": str(obj_val),
                    "confidence": conf,
                },
            )

    def ls(self, path: str = "/") -> list[dict[str, Any]]:
        """List contents of a directory path."""
        node = self._resolve(path)
        if node is None:
            return []
        if not node.is_directory:
            return [self._node_info(node)]
        return [
            self._node_info(child)
            for child in node.children.values()
        ]

    def cat(self, path: str) -> dict[str, Any] | None:
        """Read a claim node's data."""
        node = self._resolve(path)
        if node is None or node.is_directory:
            return None
        return node.claim_data

    def tree(
        self, path: str = "/", *, max_depth: int = 3,
    ) -> list[str]:
        """Return a tree-format listing."""
        node = self._resolve(path)
        if node is None:
            return []
        lines: list[str] = []
        self._tree_walk(node, "", lines, 0, max_depth)
        return lines

    def find(self, query: str) -> list[str]:
        """Find paths matching a query string."""
        query_lower = query.lower()
        matches: list[str] = []
        self._find_walk(self._root, query_lower, matches)
        return matches

    def stats(self) -> dict[str, Any]:
        dirs, claims = self._count(self._root)
        return {
            "total_directories": dirs,
            "total_claims": claims,
            "top_level_entities": len(self._root.children),
        }

    # ── Internal ─────────────────────────────────────────────────

    def _resolve(self, path: str) -> TreeNode | None:
        parts = [p for p in path.strip("/").split("/") if p]
        node = self._root
        for part in parts:
            if part not in node.children:
                return None
            node = node.children[part]
        return node

    def _node_info(self, node: TreeNode) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": node.name,
            "path": node.path,
            "type": node.node_type,
        }
        if node.is_directory:
            info["children"] = node.child_count
        if node.claim_data:
            info["confidence"] = node.claim_data.get(
                "confidence", 0
            )
        return info

    def _tree_walk(
        self,
        node: TreeNode,
        prefix: str,
        lines: list[str],
        depth: int,
        max_depth: int,
    ) -> None:
        label = node.name
        if not node.is_directory and node.claim_data:
            conf = node.claim_data.get("confidence", "?")
            label = f"{node.name} [{conf}]"
        lines.append(f"{prefix}{label}")
        if depth >= max_depth or not node.is_directory:
            return
        children = list(node.children.values())
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            self._tree_walk(
                child, prefix + connector, lines,
                depth + 1, max_depth,
            )

    def _find_walk(
        self,
        node: TreeNode,
        query: str,
        matches: list[str],
    ) -> None:
        if query in node.name.lower():
            matches.append(node.path)
        for child in node.children.values():
            self._find_walk(child, query, matches)

    def _count(
        self, node: TreeNode,
    ) -> tuple[int, int]:
        dirs = 0
        claims = 0
        if node.is_directory:
            dirs += 1
        else:
            claims += 1
        for child in node.children.values():
            d, c = self._count(child)
            dirs += d
            claims += c
        return dirs, claims
