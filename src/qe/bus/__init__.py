from qe.bus.memory_bus import MemoryBus

BUS_SINGLETON = MemoryBus()


def get_bus() -> MemoryBus:
    return BUS_SINGLETON
