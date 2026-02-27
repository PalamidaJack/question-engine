from qe.models.genome import Blueprint
from qe.runtime.service import BaseService


class ServiceRegistry:
    def __init__(self) -> None:
        self._services: dict[str, tuple[Blueprint, BaseService]] = {}

    def register(self, blueprint: Blueprint, service: BaseService) -> None:
        self._services[blueprint.service_id] = (blueprint, service)

    def get(self, service_id: str) -> BaseService | None:
        item = self._services.get(service_id)
        return item[1] if item else None

    def all_services(self) -> list[BaseService]:
        return [svc for _, svc in self._services.values()]

    def all_blueprints(self) -> list[Blueprint]:
        return [bp for bp, _ in self._services.values()]
