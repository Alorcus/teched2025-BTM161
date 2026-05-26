import logging

from .shared_components import OrderStatus, Order
from .order_store import save_order

logger = logging.getLogger("coffee_shop.state_machine")


class InvalidTransitionError(Exception):
    """Raised when an order state transition is not allowed."""


ALLOWED_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING: {
        OrderStatus.INVENTORY_CONFIRMED,
        OrderStatus.INVENTORY_ISSUES,
        OrderStatus.CANCELLED,
    },
    OrderStatus.INVENTORY_CONFIRMED: {
        OrderStatus.IN_PREPARATION,
        OrderStatus.INVENTORY_ISSUES,
        OrderStatus.CANCELLED,
    },
    OrderStatus.INVENTORY_ISSUES: {
        OrderStatus.PENDING,
        OrderStatus.CANCELLED,
    },
    OrderStatus.IN_PREPARATION: {
        OrderStatus.COMPLETED,
        OrderStatus.PREPARATION_ERROR,
        OrderStatus.CANCELLED,
    },
    OrderStatus.PREPARATION_ERROR: {
        OrderStatus.COMPLETED,
        OrderStatus.IN_PREPARATION,
        OrderStatus.CANCELLED,
    },
    OrderStatus.COMPLETED: {
        OrderStatus.REFUNDED,
    },
    OrderStatus.REFUNDED: set(),
    OrderStatus.CANCELLED: set(),
}


class OrderStateMachine:
    """Centralized enforcer for order status transitions."""

    def __init__(self) -> None:
        self._transitions = ALLOWED_TRANSITIONS

    def is_valid_transition(
        self, from_status: OrderStatus, to_status: OrderStatus
    ) -> bool:
        return to_status in self._transitions.get(from_status, set())

    def transition(
        self, order: Order, new_status: OrderStatus, *, context: str = ""
    ) -> Order:
        """Validate and apply a status transition, then persist."""
        old_status = order.status
        if not self.is_valid_transition(old_status, new_status):
            error_msg = (
                f"[TRANSITION] Cannot transition order {order.order_id_str} "
                f"from {old_status.value} to {new_status.value}"
                f"{f' (context: {context})' if context else ''}"
            )
            logger.debug(error_msg)
            raise InvalidTransitionError(error_msg)
        order.status = new_status
        save_order(order)
        logger.info(
            f"[TRANSITION] order={order.order_id_str} "
            f"from={old_status.value} to={new_status.value}"
            f"{f' context={context}' if context else ''}"
        )
        return order


state_machine = OrderStateMachine()
