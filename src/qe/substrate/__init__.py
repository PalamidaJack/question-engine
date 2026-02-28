from pathlib import Path

from qe.substrate.belief_ledger import BeliefLedger
from qe.substrate.cold_storage import ColdStorage
from qe.substrate.embeddings import EmbeddingStore
from qe.substrate.entities import EntityResolver
from qe.substrate.magma import MultiGraphQuery


class Substrate:
    def __init__(
        self,
        db_path: str = "data/belief_ledger.db",
        cold_path: str = "cold",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.belief_ledger = BeliefLedger(db_path, Path(__file__).parent / "migrations")
        # Keep .ledger as alias for backwards compatibility
        self.ledger = self.belief_ledger
        self.cold_storage = ColdStorage(cold_path)
        self.entity_resolver = EntityResolver(db_path)
        self.embeddings = EmbeddingStore(db_path, model=embedding_model)
        # MAGMA multi-graph query (event_log wired later via set_event_log)
        self._magma = MultiGraphQuery(
            embeddings=self.embeddings,
            belief_ledger=self.belief_ledger,
            entity_resolver=self.entity_resolver,
        )

    async def initialize(self) -> None:
        await self.ledger.initialize()

    async def commit_claim(self, claim):
        # Normalize entity name before committing
        claim.subject_entity_id = await self.entity_resolver.ensure_entity(
            claim.subject_entity_id
        )
        return await self.ledger.commit_claim(claim)

    async def get_claims(self, **kwargs):
        return await self.ledger.get_claims(**kwargs)

    async def get_claim_by_id(self, claim_id: str):
        return await self.ledger.get_claim_by_id(claim_id)

    async def count_claims(self, **kwargs):
        return await self.ledger.count_claims(**kwargs)

    async def retract_claim(self, claim_id: str) -> bool:
        return await self.ledger.retract_claim(claim_id)

    async def search_full_text(self, query: str, limit: int = 20):
        return await self.ledger.search_full_text(query, limit)

    async def search_semantic(self, query: str, top_k: int = 10):
        return await self.embeddings.search(query, top_k=top_k)

    def set_event_log(self, event_log) -> None:
        """Wire the EventLog into MAGMA for temporal + causal queries."""
        self._magma = MultiGraphQuery(
            embeddings=self.embeddings,
            event_log=event_log,
            belief_ledger=self.belief_ledger,
            entity_resolver=self.entity_resolver,
        )

    def set_memory_store(self, memory_store) -> None:
        """Wire MemoryStore into MAGMA for entity memory queries."""
        self._magma = MultiGraphQuery(
            embeddings=self.embeddings,
            event_log=self._magma._event_log,
            belief_ledger=self.belief_ledger,
            entity_resolver=self.entity_resolver,
            memory_store=memory_store,
        )

    async def multi_query(self, query: str, **kwargs):
        """MAGMA multi-graph query: fan out across semantic, temporal, entity, causal."""
        return await self._magma.multi_query(query, **kwargs)
