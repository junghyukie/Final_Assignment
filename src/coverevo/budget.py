"""Fail-closed Decimal budget ledger and crash-safe request lifecycle."""

from __future__ import annotations

from decimal import Decimal
import json
import math
import os
from pathlib import Path
import threading
from typing import Any

from .storage import decode_lossless, dump_json_atomic, encode_lossless, load_json_lossless


class BudgetError(RuntimeError):
    pass


class BudgetExceededError(BudgetError):
    pass


class PendingRequestError(BudgetError):
    pass


class DuplicateReconciliationError(BudgetError):
    pass


class LedgerLockedError(BudgetError):
    pass


class LiveBudgetError(BudgetError):
    pass


_HELD_PATHS: set[str] = set()
_HELD_PATHS_GUARD = threading.Lock()


class FileLock:
    """Non-blocking one-byte OS lock with a same-process contention guard."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None
        self._identity: str | None = None

    def acquire(self) -> None:
        if self._handle is not None:
            raise LedgerLockedError(f"lock is already held: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        identity = str(self.path.resolve())
        with _HELD_PATHS_GUARD:
            if identity in _HELD_PATHS:
                raise LedgerLockedError(f"ledger lock is held: {self.path}")
            _HELD_PATHS.add(identity)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            handle.close()
            with _HELD_PATHS_GUARD:
                _HELD_PATHS.discard(identity)
            raise LedgerLockedError(f"ledger lock is held: {self.path}") from error
        self._handle = handle
        self._identity = identity

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
            if self._identity is not None:
                with _HELD_PATHS_GUARD:
                    _HELD_PATHS.discard(self._identity)
            self._identity = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def _valid_decimal(value: Decimal, *, positive: bool) -> bool:
    if not isinstance(value, Decimal) or not value.is_finite():
        return False
    return value > 0 if positive else value >= 0


class BudgetLedger:
    """Append-only financial state; JSONL replay is authoritative."""

    def __init__(self, directory: str | Path, *, cap: Decimal) -> None:
        if not _valid_decimal(cap, positive=True):
            raise ValueError("budget cap must be a positive finite Decimal")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_path = self.directory / "ledger.jsonl"
        self.snapshot_path = self.directory / "ledger_state.json"
        self.lock_path = self.directory / "ledger.lock"
        self.cap = cap
        self.events: list[dict[str, Any]] = []
        self.spent = Decimal("0")
        self.reservations: dict[str, Decimal] = {}
        self.reservation_metadata: dict[str, dict[str, Any]] = {}
        self.unknown_billing: set[str] = set()
        self.reconciled: set[str] = set()
        self.next_request_number = 1
        self.over_cap = False
        with FileLock(self.lock_path):
            if self.log_path.exists() and self.log_path.stat().st_size:
                self._recover_locked()
            else:
                self._reset()
                self._append_locked("LEDGER_CREATED", cap=cap)

    def _reset(self) -> None:
        self.events = []
        self.spent = Decimal("0")
        self.reservations = {}
        self.reservation_metadata = {}
        self.unknown_billing = set()
        self.reconciled = set()
        self.next_request_number = 1
        self.over_cap = False

    def _apply(self, event: dict[str, Any]) -> None:
        kind = event["event"]
        if kind == "LEDGER_CREATED":
            recorded_cap = event["cap"]
            if recorded_cap != self.cap:
                raise ValueError("budget ledger cap mismatch")
            return
        request_id = event.get("request_id")
        if kind == "RESERVED":
            if request_id in self.reservations or request_id in self.reconciled:
                raise ValueError("duplicate reservation in ledger")
            self.reservations[request_id] = event["estimate"]
            self.reservation_metadata[request_id] = dict(event.get("metadata", {}))
            self.next_request_number = max(
                self.next_request_number, event["request_number"] + 1
            )
        elif kind == "RECONCILED":
            if request_id not in self.reservations:
                raise ValueError("reconciliation has no matching reservation")
            self.reservations.pop(request_id)
            self.reservation_metadata.pop(request_id, None)
            self.spent += event["actual_cost"]
            self.reconciled.add(request_id)
        elif kind == "UNKNOWN_BILLING":
            if request_id not in self.reservations:
                raise ValueError("unknown billing has no matching reservation")
            self.reservations.pop(request_id)
            self.reservation_metadata.pop(request_id, None)
            self.unknown_billing.add(request_id)
        elif kind == "BILLING_RESOLVED":
            if request_id not in self.unknown_billing:
                raise ValueError("billing resolution has no unknown request")
            self.unknown_billing.remove(request_id)
            self.spent += event["actual_cost"]
            self.reconciled.add(request_id)
        elif kind in {"POST_STARTED", "RESPONSE_RECEIVED", "CHECKPOINTED"}:
            return
        else:
            raise ValueError(f"unknown budget event: {kind}")
        self.over_cap = self.spent > self.cap

    def _recover_locked(self) -> None:
        self._reset()
        expected_sequence = 1
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = decode_lossless(json.loads(line))
                if event.get("sequence") != expected_sequence:
                    raise ValueError("budget ledger sequence is not contiguous")
                expected_sequence += 1
                self._apply(event)
                self.events.append(event)
        if not self.events or self.events[0]["event"] != "LEDGER_CREATED":
            raise ValueError("budget ledger is missing creation event")
        self.over_cap = self.spent > self.cap
        self._persist_snapshot_locked()

    def refresh(self) -> None:
        with FileLock(self.lock_path):
            self._recover_locked()

    def _append_locked(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {"sequence": len(self.events) + 1, "event": kind, **fields}
        encoded = json.dumps(
            encode_lossless(event),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._apply(event)
        self.events.append(event)
        self._persist_snapshot_locked()
        return event

    def _persist_snapshot_locked(self) -> None:
        dump_json_atomic(
            self.snapshot_path,
            {
                "cap": self.cap,
                "spent": self.spent,
                "reservations": self.reservations,
                "reservation_metadata": self.reservation_metadata,
                "unknown_billing": tuple(sorted(self.unknown_billing)),
                "reconciled": tuple(sorted(self.reconciled)),
                "next_request_number": self.next_request_number,
                "over_cap": self.over_cap,
                "last_sequence": len(self.events),
            },
        )

    def reserve(
        self, estimate: Decimal, *, metadata: dict[str, Any] | None = None
    ) -> str:
        if not _valid_decimal(estimate, positive=True):
            raise ValueError("reservation estimate must be a positive finite Decimal")
        with FileLock(self.lock_path):
            self._recover_locked()
            if self.unknown_billing:
                raise PendingRequestError("unknown billing must be reconciled first")
            if self.reservations:
                raise PendingRequestError("another request reservation is unresolved")
            if self.over_cap or self.spent + estimate > self.cap:
                raise BudgetExceededError("reservation would exceed local cap")
            number = self.next_request_number
            request_id = f"request-{number:08d}"
            self._append_locked(
                "RESERVED",
                request_id=request_id,
                request_number=number,
                estimate=estimate,
                metadata=dict(metadata or {}),
            )
            return request_id

    def reconcile(self, request_id: str, actual_cost: Decimal | None) -> None:
        if actual_cost is not None and not _valid_decimal(actual_cost, positive=False):
            raise ValueError("actual cost must be a non-negative finite Decimal or None")
        with FileLock(self.lock_path):
            self._recover_locked()
            if request_id in self.reconciled or request_id in self.unknown_billing:
                raise DuplicateReconciliationError(
                    f"request already reconciled: {request_id}"
                )
            if request_id not in self.reservations:
                raise KeyError(f"unknown reservation: {request_id}")
            metadata = dict(self.reservation_metadata.get(request_id, {}))
            if actual_cost is None:
                self._append_locked(
                    "UNKNOWN_BILLING",
                    request_id=request_id,
                    metadata=metadata,
                )
            else:
                self._append_locked(
                    "RECONCILED",
                    request_id=request_id,
                    actual_cost=actual_cost,
                    metadata=metadata,
                )

    def resolve_unknown(
        self, request_id: str, actual_cost: Decimal, *, evidence: str
    ) -> None:
        """Resolve uncertainty once using independently verified billing evidence."""
        if not _valid_decimal(actual_cost, positive=False):
            raise ValueError("actual cost must be a non-negative finite Decimal")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("billing resolution requires evidence")
        with FileLock(self.lock_path):
            self._recover_locked()
            if request_id not in self.unknown_billing:
                raise DuplicateReconciliationError("request is not awaiting billing resolution")
            self._append_locked(
                "BILLING_RESOLVED", request_id=request_id,
                actual_cost=actual_cost, evidence=evidence,
            )

    def record_event(
        self, kind: str, request_id: str, *, details: dict[str, Any] | None = None
    ) -> None:
        if kind not in {"POST_STARTED", "RESPONSE_RECEIVED", "CHECKPOINTED"}:
            raise ValueError("unsupported request lifecycle event")
        with FileLock(self.lock_path):
            self._recover_locked()
            if kind == "CHECKPOINTED":
                known = request_id in self.reconciled
            else:
                known = request_id in self.reservations
            if not known:
                raise KeyError(f"request is not in the required ledger state: {request_id}")
            metadata = dict(self.reservation_metadata.get(request_id, {}))
            self._append_locked(
                kind,
                request_id=request_id,
                metadata=metadata,
                details=dict(details or {}),
            )


def reserve_request(ledger: BudgetLedger, estimate: Decimal) -> str:
    return ledger.reserve(estimate)


def reconcile_request(
    ledger: BudgetLedger, request_id: str, actual_cost: Decimal | None
) -> None:
    ledger.reconcile(request_id, actual_cost)


def validate_live_budget(
    allow_live: bool,
    initial_verified_allowance: Decimal | None,
    local_cap: Decimal | None,
) -> Decimal:
    """Return the usable planning cap only when all live prerequisites are explicit."""
    if not allow_live:
        raise LiveBudgetError("live access has not been approved")
    if initial_verified_allowance is None or not _valid_decimal(
        initial_verified_allowance, positive=True
    ):
        raise LiveBudgetError("verified finite key allowance is required")
    if local_cap is None or not _valid_decimal(local_cap, positive=True):
        raise LiveBudgetError("a positive finite local cap is required")
    return min(local_cap, initial_verified_allowance * Decimal("0.80"))


class RequestLifecycle:
    """Persist request boundaries so resume never guesses whether to resend POST."""

    _PHASE_BY_EVENT = {
        "RESERVED": "RESERVED",
        "POST_STARTED": "POST_STARTED",
        "RESPONSE_RECEIVED": "RESPONSE_RECEIVED",
        "RECONCILED": "RECONCILED",
        "UNKNOWN_BILLING": "UNKNOWN_BILLING",
        "BILLING_RESOLVED": "RECONCILED",
        "CHECKPOINTED": "CHECKPOINTED",
    }

    def __init__(
        self,
        directory: str | Path,
        ledger: BudgetLedger,
        *,
        context_hash: str,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "request_state.json"
        self.ledger = ledger
        self.context_hash = context_hash
        self.state: dict[str, Any] | None = None
        if self.state_path.exists():
            loaded = load_json_lossless(self.state_path)
            if not isinstance(loaded, dict):
                raise PendingRequestError("request state is malformed")
            self.state = loaded
            if self.state.get("context_hash") != context_hash:
                raise PendingRequestError("request context hash mismatch")
        self.ledger.refresh()
        if self.state is not None:
            self._synchronize_from_log()
        if self.ledger.reservations:
            request_id = next(iter(self.ledger.reservations))
            metadata = self.ledger.reservation_metadata.get(request_id, {})
            if metadata.get("context_hash") != context_hash:
                if self.state is None:
                    raise PendingRequestError("request context hash mismatch")
            elif self.state is None or self.state.get('request_id') != request_id:
                if self.state is not None and (
                    self.state.get('phase') != 'CHECKPOINTED'
                    or metadata.get('slot', 0) <= self.state.get('slot', 0)
                ):
                    raise PendingRequestError('new reservation conflicts with existing request state')
                # Recover the append-before-state-write crash boundary, including
                # branches whose previous slot was already checkpointed.
                self.state = {
                    "context_hash": context_hash,
                    "slot": metadata.get("slot"),
                    "request_id": request_id,
                    "phase": "RESERVED",
                    "estimate": self.ledger.reservations[request_id],
                }
                self._save()
        self._synchronize_from_log()

    @property
    def phase(self) -> str | None:
        return None if self.state is None else self.state.get("phase")

    def _save(self) -> None:
        if self.state is not None:
            dump_json_atomic(self.state_path, self.state)

    def _synchronize_from_log(self) -> None:
        if self.state is None:
            return
        request_id = self.state["request_id"]
        matching = [
            event for event in self.ledger.events if event.get("request_id") == request_id
        ]
        if not matching:
            raise PendingRequestError("request state has no ledger events")
        latest = matching[-1]
        phase = self._PHASE_BY_EVENT.get(latest["event"])
        if phase is not None and phase != self.state.get("phase"):
            self.state["phase"] = phase
            if latest["event"] == "RESPONSE_RECEIVED":
                details = latest.get("details", {})
                self.state["response_id"] = details.get("response_id")
                self.state["actual_cost"] = details.get("actual_cost")
            elif latest['event'] == 'BILLING_RESOLVED':
                self.state['actual_cost'] = latest['actual_cost']
            self._save()

    def start_slot(self, slot: int, estimate: Decimal) -> str:
        if type(slot) is not int or slot < 1:
            raise ValueError("slot must be a positive integer")
        if self.state is not None:
            phase = self.state["phase"]
            same_slot = self.state.get("slot") == slot
            if same_slot and phase == "RESERVED":
                if self.state.get("estimate") != estimate:
                    raise PendingRequestError("reservation estimate changed on resume")
                return self.state["request_id"]
            if phase in {"POST_STARTED", "RESPONSE_RECEIVED", "UNKNOWN_BILLING"}:
                raise PendingRequestError("posted slot must not be resent")
            if phase == "RECONCILED":
                raise PendingRequestError("reconciled request must be checkpointed first")
            if phase == "CHECKPOINTED" and slot <= self.state.get("slot", 0):
                raise PendingRequestError("slot is already checkpointed")
        request_id = self.ledger.reserve(
            estimate,
            metadata={
                "slot": slot,
                "context_hash": self.context_hash,
            },
        )
        self.state = {
            "context_hash": self.context_hash,
            "slot": slot,
            "request_id": request_id,
            "phase": "RESERVED",
            "estimate": estimate,
        }
        self._save()
        return request_id

    def _require(self, request_id: str, phase: str) -> None:
        if (
            self.state is None
            or self.state.get("request_id") != request_id
            or self.state.get("phase") != phase
        ):
            raise PendingRequestError(
                f"request must be in {phase} phase and must not be resent"
            )

    def mark_post_started(self, request_id: str) -> None:
        self._require(request_id, "RESERVED")
        self.ledger.record_event(
            "POST_STARTED", request_id, details={"slot": self.state["slot"]}
        )
        self.state["phase"] = "POST_STARTED"
        self._save()

    def record_response(
        self,
        request_id: str,
        *,
        response_id: str | None,
        actual_cost: Decimal | None,
    ) -> None:
        self._require(request_id, "POST_STARTED")
        if actual_cost is not None and not _valid_decimal(actual_cost, positive=False):
            raise ValueError("actual cost must be a non-negative finite Decimal or None")
        self.ledger.record_event(
            "RESPONSE_RECEIVED",
            request_id,
            details={
                "slot": self.state["slot"],
                "response_id": response_id,
                "actual_cost": actual_cost,
            },
        )
        self.state.update(
            {
                "phase": "RESPONSE_RECEIVED",
                "response_id": response_id,
                "actual_cost": actual_cost,
            }
        )
        self._save()

    def reconcile_response(self, request_id: str) -> None:
        self._require(request_id, "RESPONSE_RECEIVED")
        actual_cost = self.state.get("actual_cost")
        self.ledger.reconcile(request_id, actual_cost)
        self.state["phase"] = (
            "UNKNOWN_BILLING" if actual_cost is None else "RECONCILED"
        )
        self._save()

    def checkpoint(self, request_id: str) -> None:
        self._require(request_id, "RECONCILED")
        self.ledger.record_event(
            "CHECKPOINTED", request_id, details={"slot": self.state["slot"]}
        )
        self.state["phase"] = "CHECKPOINTED"
        self._save()

    def post_count(self, slot: int) -> int:
        self.ledger.refresh()
        return sum(
            event["event"] == "POST_STARTED"
            and (
                event.get("metadata", {}).get("slot") == slot
                or event.get("details", {}).get("slot") == slot
            )
            for event in self.ledger.events
        )
