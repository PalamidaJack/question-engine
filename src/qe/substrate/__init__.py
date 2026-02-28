from pathlib import Path

from qe.substrate.belief_ledger import BeliefLedger
from qe.substrate.cold_storage import ColdStorage
from qe.substrate.embeddings import EmbeddingStore
from qe.substrate.entities import EntityResolver


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
