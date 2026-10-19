"""Resumable OpenRouter transport and usage accounting for the VISTA judge."""

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .provenance import atomic_json, digest, read_jsonl

BASE_URL = "https://openrouter.ai/api/v1"


def read_keys(paths):
    keys = [Path(p).read_text().strip() for p in paths]
    if not keys or any(
        not k.startswith("sk-or-") or any(c.isspace() for c in k) for k in keys
    ):
        raise ValueError("Key files must contain one OpenRouter API key each")
    return list(dict.fromkeys(keys))


class Client:
    def __init__(self, keys, budget, ledger, timeout=120):
        self.keys, self.budget, self.ledger, self.timeout = (
            keys,
            budget,
            Path(ledger),
            timeout,
        )
        self.lock = threading.Lock()
        self.cooldown = 0.0
        self.stop = False
        self.spent = 0.0
        self.billed_items = set()
        self.inflight = self.ledger.with_suffix(".inflight")
        self.inflight.mkdir(exist_ok=True)
        if any(self.inflight.iterdir()):
            raise ValueError(
                "Interrupted API request; reconcile billing before resuming"
            )
        if self.ledger.exists():
            for row in read_jsonl(self.ledger):
                if row.get("cost") is None:
                    raise ValueError(
                        "An earlier request has unknown cost; reconcile usage before resuming"
                    )
                self.spent += row["cost"]
                self.billed_items.add(row["item_id"])

    def complete(self, payload, key_index, item_id):
        if item_id in self.billed_items:
            raise ValueError(
                "Item already billed but response unavailable; reconcile before retry"
            )
        for attempt in range(5):
            with self.lock:
                if self.stop or self.spent >= self.budget:
                    raise RuntimeError(
                        "Judge stopped at spending limit or earlier request failure"
                    )
                delay = max(0.0, self.cooldown - time.monotonic())
            if delay:
                time.sleep(delay)
            request = urllib.request.Request(
                BASE_URL + "/chat/completions",
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": "Bearer " + self.keys[key_index % len(self.keys)],
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            pending = self.inflight / (digest(item_id) + ".json")
            atomic_json(pending, {"item_id": item_id, "status": "request_pending"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
            except urllib.error.HTTPError as exc:
                pending.unlink()
                # Back off globally, using the same key. Extra keys do not bypass account limits.
                if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                    try:
                        retry = float(exc.headers.get("Retry-After", 0))
                    except ValueError:
                        retry = 0
                    with self.lock:
                        self.cooldown = max(
                            self.cooldown,
                            time.monotonic() + max(retry, 2 ** (attempt + 1)),
                        )
                    continue
                with self.lock:
                    self.stop = True
                raise RuntimeError(
                    f"OpenRouter HTTP {exc.code}; response body withheld"
                ) from None
            except Exception:
                self.record(item_id, None, None)
                pending.unlink()
                with self.lock:
                    self.stop = True
                raise RuntimeError(
                    "OpenRouter transport failure; reconcile possible billing before retry"
                ) from None
            usage = data.get("usage", {})
            cost = usage.get("cost")
            if (
                not isinstance(cost, (int, float))
                or not math.isfinite(cost)
                or cost < 0
            ):
                cost = None
            self.record(item_id, cost, data.get("id"))
            pending.unlink()
            if cost is None or data.get("error"):
                with self.lock:
                    self.stop = True
                raise RuntimeError(
                    "OpenRouter response lacks valid cost or contains an error; inspect request ID"
                )
            return data
        raise RuntimeError("OpenRouter retry limit reached")

    def record(self, item_id, cost, request_id):
        with self.lock:
            if cost is not None:
                self.spent += cost
            with self.ledger.open("a") as writer:
                writer.write(
                    json.dumps(
                        {"item_id": item_id, "cost": cost, "request_id": request_id}
                    )
                    + "\n"
                )
                writer.flush()
                os.fsync(writer.fileno())
