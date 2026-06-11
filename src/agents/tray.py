import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger("coffee_shop.tray")

_trays: dict[str, list["TrayEntry"]] = {}


@dataclass
class TrayEntry:
    item_name: str
    quantity: int
    category: str
    contaminated: bool = False


def place_on_tray(
    order_id: str,
    item_name: str,
    quantity: int,
    category: str,
    contaminated: bool = False,
) -> list[dict]:
    entry = TrayEntry(
        item_name=item_name,
        quantity=quantity,
        category=category,
        contaminated=contaminated,
    )
    if order_id not in _trays:
        _trays[order_id] = []
    _trays[order_id].append(entry)
    logger.debug(f"Item placed on tray for order {order_id}: {entry}")
    return tray_as_list(order_id)


def get_tray(order_id: str) -> list[TrayEntry]:
    return _trays.get(order_id, [])


def clear_tray(order_id: str) -> None:
    logger.debug(f"Clearing tray for order {order_id}")
    _trays.pop(order_id, None)


def tray_as_list(order_id: str) -> list[dict]:
    return [asdict(e) for e in get_tray(order_id)]
