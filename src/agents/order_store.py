import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("coffee_shop.order_store")

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import event
from sqlalchemy.orm import make_transient
from sqlmodel import Session, SQLModel, create_engine, select

from .shared_components import Order, MenuItem, MENU

_DB_PATH = Path(__file__).resolve().parents[2] / "coffee_shop.db"
_write_lock = threading.Lock()

engine = create_engine(
    f"sqlite:///{_DB_PATH}",
    connect_args={"check_same_thread": False, "timeout": 10},
    echo=False,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist and seed inventory from MENU."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        first = session.exec(select(MenuItem)).first()
        if first is None:
            for key, item in MENU.items():
                session.add(MenuItem(
                    name=key, price=item.price, stock=item.stock, category=item.category,
                ))
            session.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_order_id(order_id: str) -> Optional[int]:
    """Parse 'ORD0042' -> 42, or return None on bad format."""
    try:
        return int(order_id.upper().replace("ORD", ""))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Order CRUD
# ---------------------------------------------------------------------------

def save_order(order: Order) -> None:
    """Persist an Order (insert or update)."""
    # may want to check for allowed status transitions here in the future
    is_new = order.id is None
    if not is_new:
        from datetime import datetime, timezone
        order.last_modified = datetime.now(timezone.utc)
    with _write_lock:
        with Session(engine) as session:
            if not is_new:
                merged = session.merge(order)
                session.commit()
                session.refresh(merged)
                # Copy back the DB-assigned state to the caller's object
                order.id = merged.id
            else:
                session.add(order)
                session.commit()
                session.refresh(order)
    if is_new:
        logger.debug("Order %s created for %s — %d item(s), $%.2f", order.order_id_str, order.customer, len(order.items), order.total)
    else:
        logger.debug("Order %s updated — status=%s, total=$%.2f", order.order_id_str, order.status.value, order.total)


def load_order(order_id: str) -> Optional[Order]:
    """Load an Order by ID string (e.g. 'ORD0001'). Returns None if not found."""
    int_id = _parse_order_id(order_id)
    if int_id is None:
        return None
    with Session(engine) as session:
        order = session.get(Order, int_id)
        if order is None:
            return None
        _ = order.items  # force-load relationship while session is open
        session.expunge(order)
        make_transient(order)
        for item in order.items:
            make_transient(item)
        return order


# ---------------------------------------------------------------------------
# Inventory operations
# ---------------------------------------------------------------------------

def check_inventory_availability(order: Order) -> dict:
    """Check stock levels for every item in the order (read-only)."""
    with Session(engine) as session:
        details = []
        all_available = True
        unavailable_items = []

        for oi in order.items:
            item = session.get(MenuItem, oi.name)
            if item is None:
                return {"error": f"Item '{oi.name}' not found in inventory."}
            avail = item.stock
            if avail >= oi.quantity:
                status = "available"
            elif avail > 0:
                status = "partial"
                all_available = False
                unavailable_items.append(oi.name)
            else:
                status = "out_of_stock"
                all_available = False
                unavailable_items.append(oi.name)
            details.append({
                "name": oi.name,
                "requested": oi.quantity,
                "available": avail,
                "status": status,
            })

        return {
            "all_available": all_available,
            "details": details,
            "unavailable_items": unavailable_items,
        }


def check_and_update_stock(order: Order) -> list[dict]:
    """Atomically check availability and deduct stock for every order item.

    Must be called only when order.status == 'inventory_confirmed'.
    Raises KeyError if an item is missing, ValueError on insufficient stock.
    """
    items_report = []
    with _write_lock:
        with Session(engine) as session:
            for oi in order.items:
                item = session.get(MenuItem, oi.name)
                if item is None:
                    raise KeyError(f"Item '{oi.name}' not found in inventory.")
                stock_before = item.stock
                if stock_before < oi.quantity:
                    raise ValueError(
                        f"Insufficient stock for '{oi.name}': "
                        f"need {oi.quantity}, have {stock_before}"
                    )
                item.stock -= oi.quantity
                items_report.append({
                    "name": oi.name,
                    "quantity_removed": oi.quantity,
                    "previous_stock": stock_before,
                    "new_stock": stock_before - oi.quantity,
                })
            session.commit()
    if items_report:
        deductions = ", ".join(f"{r['name']} {r['previous_stock']}->{r['new_stock']}" for r in items_report)
        logger.debug("Stock deducted for %s: %s", order.order_id_str, deductions)
    return items_report


def reset_inventory() -> None:
    """Reset all inventory stock levels to MENU defaults."""
    with _write_lock:
        with Session(engine) as session:
            for key, defaults in MENU.items():
                item = session.get(MenuItem, key)
                if item:
                    item.stock = defaults.stock
                else:
                    session.add(MenuItem(
                        name=key, price=defaults.price, stock=defaults.stock, category=defaults.category,
                    ))
            session.commit()


def set_item_stock(name: str, stock: int) -> None:
    """Set stock for a single inventory item (used by scenario buttons)."""
    with _write_lock:
        with Session(engine) as session:
            item = session.get(MenuItem, name)
            if item:
                item.stock = stock
                session.commit()


def get_all_inventory() -> dict[str, MenuItem]:
    """Return full inventory as {name: MenuItem}. Used by the UI."""
    with Session(engine) as session:
        items = session.exec(select(MenuItem)).all()
        result = {}
        for item in items:
            session.expunge(item)
            result[item.name] = item
        return result


def get_inventory_item(name: str) -> Optional[dict]:
    """Return a single inventory item as a dict, or None."""
    with Session(engine) as session:
        item = session.get(MenuItem, name)
        if item is None:
            return None
        return item.model_dump()


def get_alternatives_from_db(item_name: str) -> list[dict]:
    """Get in-stock alternatives in the same category."""
    with Session(engine) as session:
        item = session.get(MenuItem, item_name)
        if item is None:
            return []
        statement = select(MenuItem).where(
            MenuItem.category == item.category,
            MenuItem.name != item_name,
            MenuItem.stock > 0,
        )
        alts = session.exec(statement).all()
        return [{"name": a.name, "price": a.price, "stock": a.stock} for a in alts]


# ---------------------------------------------------------------------------
# Shared LangChain tool — all agents get this
# ---------------------------------------------------------------------------

class GetOrderSchema(BaseModel):
    order_id: str = Field(description="The order ID to look up (e.g. 'ORD0001')")


@tool(args_schema=GetOrderSchema)
def get_order(order_id: str) -> str:
    """Look up the current state of an order by its ID."""
    order = load_order(order_id)
    if order is None:
        return f"Error: Order '{order_id}' not found."

    summary = f"Order {order.order_id_str} ({order.status.value}) for {order.customer}:\n"
    for item in order.items:
        extras_str = f" with {', '.join(item.extras)}" if item.extras else ""
        size_str = f" ({item.size.value})" if item.size else ""
        summary += f"  - {item.quantity}x {item.name.title()}{size_str}{extras_str}: ${item.price:.2f}\n"
    summary += f"Total: ${order.total:.2f}"
    return summary
