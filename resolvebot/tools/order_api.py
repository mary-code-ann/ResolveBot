"""Mock Order API backed by a local JSON file."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from resolvebot.config import MAX_RETRIES, ORDERS_PATH, ORDERS_SEED_PATH

RETURN_WINDOW_DAYS = 30


def _load_orders() -> dict[str, Any]:
    if not ORDERS_PATH.exists():
        reset_orders()
    with ORDERS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_orders(orders: dict[str, Any]) -> None:
    ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ORDERS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(orders, handle, indent=2)


def reset_orders() -> None:
    """Restore demo orders from the seed file."""
    if ORDERS_SEED_PATH.exists():
        shutil.copyfile(ORDERS_SEED_PATH, ORDERS_PATH)
        return
    ORDERS_PATH.write_text("{}", encoding="utf-8")


def _with_retries(fn):
    last_error: Exception | None = None
    for _ in range(MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.15)
    raise last_error or RuntimeError("Order API failed")


def get_order(order_id: str) -> dict[str, Any]:
    """Return one order record, or a not_found payload."""

    def _run() -> dict[str, Any]:
        orders = _load_orders()
        key = order_id.strip().upper()
        order = orders.get(key)
        if not order:
            return {
                "ok": False,
                "error": "not_found",
                "order_id": key,
                "message": f"No order exists for {key}.",
            }
        days = order.get("delivered_days_ago")
        eligible = (
            order.get("status") == "delivered"
            and isinstance(days, int)
            and days <= RETURN_WINDOW_DAYS
        )
        return {
            "ok": True,
            "order": order,
            "within_return_window": eligible,
            "return_window_days": RETURN_WINDOW_DAYS,
        }

    return _with_retries(_run)


def cancel_order(order_id: str) -> dict[str, Any]:
    """Cancel if policy allows; otherwise explain why not."""

    def _run() -> dict[str, Any]:
        orders = _load_orders()
        key = order_id.strip().upper()
        order = orders.get(key)
        if not order:
            return {
                "ok": False,
                "error": "not_found",
                "order_id": key,
                "message": f"No order exists for {key}.",
            }

        status = order.get("status")
        days = order.get("delivered_days_ago")

        if status == "cancelled":
            return {
                "ok": False,
                "error": "already_cancelled",
                "order_id": key,
                "order": order,
                "message": f"{key} is already cancelled.",
            }

        if status == "shipped":
            return {
                "ok": False,
                "error": "in_transit",
                "order_id": key,
                "order": order,
                "message": (
                    f"{key} has shipped and is not delivered yet. "
                    "A refund cannot be issued until delivery, unless a human "
                    "attempts a carrier intercept."
                ),
            }

        if status == "processing":
            order["status"] = "cancelled"
            orders[key] = order
            _save_orders(orders)
            return {
                "ok": True,
                "order_id": key,
                "order": order,
                "message": f"{key} was cancelled before shipment. A full refund will post in 5–10 business days.",
            }

        if status == "delivered":
            if not isinstance(days, int) or days > RETURN_WINDOW_DAYS:
                return {
                    "ok": False,
                    "error": "outside_window",
                    "order_id": key,
                    "order": order,
                    "message": (
                        f"{key} was delivered {days} days ago, outside the "
                        f"{RETURN_WINDOW_DAYS}-day return window. "
                        "A change-of-mind cancel is not allowed. Warranty claims go to a human."
                    ),
                }
            order["status"] = "cancelled"
            orders[key] = order
            _save_orders(orders)
            return {
                "ok": True,
                "order_id": key,
                "order": order,
                "message": (
                    f"{key} is within the {RETURN_WINDOW_DAYS}-day window and was cancelled. "
                    "Return label next; refund posts 5–10 business days after the warehouse scan."
                ),
            }

        return {
            "ok": False,
            "error": "unknown_status",
            "order_id": key,
            "order": order,
            "message": f"{key} has status '{status}', which this mock API cannot cancel.",
        }

    return _with_retries(_run)


def seed_path() -> Path:
    return ORDERS_SEED_PATH
