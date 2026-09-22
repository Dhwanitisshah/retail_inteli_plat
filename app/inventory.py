"""
Inventory adapter (mock). The HTML reference demo ("The Physical Ledger")
triangulates shelf availability against a stock-room app's backroom count
and an ERP's system-wide count to decide whether a void is a restock job,
a purchasing problem, or phantom stock. This repo has no real stock-room/ERP
integration, so `MockInventoryAdapter` stands in for one behind the same
interface a real integration (REST/EDI/whatever the store's WMS speaks)
would implement -- swap the implementation, not the callers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict


class InventoryAdapter(ABC):
    @abstractmethod
    def get_backroom_units(self, sku_id: str) -> int:
        """Units of this SKU currently in the backroom (not on the shelf)."""

    @abstractmethod
    def get_system_units(self, sku_id: str) -> int:
        """What the ERP/system-of-record believes total on-hand stock is
        (shelf + backroom, in principle). Divergence from shelf+backroom is
        phantom stock -- present but unaccounted for, or absent but still
        "in stock" on paper."""

    @abstractmethod
    def transfer_to_shelf(self, sku_id: str, requested_units: int) -> int:
        """Move up to `requested_units` from backroom to shelf; returns the
        units actually moved (bounded by what's available)."""


@dataclass
class MockInventoryAdapter(InventoryAdapter):
    """In-memory backroom/system counts, seeded from store_layout.json.
    Every mutation happens through this class so it's a drop-in replacement
    point for a real WMS/ERP client later."""
    backroom: Dict[str, int] = field(default_factory=dict)
    system: Dict[str, int] = field(default_factory=dict)

    def get_backroom_units(self, sku_id: str) -> int:
        return self.backroom.get(sku_id, 0)

    def get_system_units(self, sku_id: str) -> int:
        return self.system.get(sku_id, self.backroom.get(sku_id, 0))

    def transfer_to_shelf(self, sku_id: str, requested_units: int) -> int:
        available = self.backroom.get(sku_id, 0)
        moved = max(0, min(available, requested_units))
        self.backroom[sku_id] = available - moved
        return moved
