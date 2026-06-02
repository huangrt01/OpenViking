# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Operation admission hooks that run after extraction and before apply.

Admission adapters may redirect a create proposal to an existing memory before
the operation reaches ``MemoryUpdater``. The base layer owns lifecycle concerns
such as scope locking, candidate refresh, decision envelopes, and link URI
rewrites; memory-type-specific adapters own semantic matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

from openviking.server.identity import RequestContext
from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation, ResolvedOperations
from openviking.storage.viking_fs import VikingFS
from openviking.telemetry import get_current_telemetry, tracer


@dataclass
class AdmissionScope:
    """Candidate scope refreshed under an admission lock."""

    memory_type: str
    operation_uri: Optional[str] = None
    parent_uri: Optional[str] = None
    lock_uri: Optional[str] = None
    candidate_uris: List[str] = field(default_factory=list)


@dataclass
class AdmissionDecision:
    """Adapter decision for one extracted operation."""

    action: str
    reason: str
    confidence: float = 0.0
    target_uri: Optional[str] = None
    target_memory_file: Optional[MemoryFile] = None
    candidate_uris: List[str] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


class OperationAdmissionAdapter:
    """Base contract for memory-type-specific operation admission."""

    def supports(self, operation: ResolvedOperation, schema: Any, ctx: RequestContext) -> bool:
        return False

    async def derive_scope(
        self,
        operation: ResolvedOperation,
        schema: Any,
        provider: Any,
        ctx: RequestContext,
        viking_fs: VikingFS,
    ) -> AdmissionScope:
        raise NotImplementedError

    async def refresh_candidates(
        self,
        scope: AdmissionScope,
        ctx: RequestContext,
        viking_fs: VikingFS,
        provider: Any = None,
    ) -> list[MemoryFile]:
        raise NotImplementedError

    async def decide(
        self,
        operation: ResolvedOperation,
        candidates: list[MemoryFile],
        scope: AdmissionScope,
    ) -> AdmissionDecision:
        return AdmissionDecision(
            action="allow_create",
            reason="adapter_no_decision",
            candidate_uris=[candidate.uri for candidate in candidates],
        )

    def apply_decision(
        self,
        operation: ResolvedOperation,
        decision: AdmissionDecision,
    ) -> tuple[list[str], Optional[str]]:
        return [], None


def _get_schema(registry: Any, memory_type: str) -> Any:
    getter = getattr(registry, "get", None)
    if callable(getter):
        return getter(memory_type)
    if isinstance(registry, dict):
        return registry.get(memory_type)
    return None


def _value_shape(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        if isinstance(value.get("blocks"), list):
            return "dict_blocks"
        return "dict"
    return type(value).__name__


def _merge_op_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return getattr(value, "value", None) or str(value)


def _schema_merge_ops(schema: Any) -> dict[str, Optional[str]]:
    return {
        getattr(field, "name", ""): _merge_op_name(getattr(field, "merge_op", None))
        for field in (getattr(schema, "fields", None) or [])
        if getattr(field, "name", "")
    }


def _operation_field_traces(
    operation: ResolvedOperation,
    schema: Any,
) -> list[dict[str, Any]]:
    merge_ops = _schema_merge_ops(schema)
    return [
        {
            "field": field_name,
            "merge_op": merge_ops.get(field_name),
            "input_shape": _value_shape(field_value),
            "wrapper_shape": None,
        }
        for field_name, field_value in sorted((operation.memory_fields or {}).items())
    ]


def _build_admission_trace(
    *,
    operation: ResolvedOperation,
    scope: AdmissionScope,
    candidates: list[MemoryFile],
    decision: AdmissionDecision,
    original_uris: list[str],
    input_shape: str,
    fields: list[dict[str, Any]],
    output_uris: list[str],
    applied_target_uri: Optional[str],
) -> dict[str, Any]:
    return {
        "trace_type": "admission_decision",
        "uri": original_uris[0] if original_uris else None,
        "uris": original_uris,
        "output_uris": output_uris,
        "memory_type": operation.memory_type,
        "input_shape": input_shape,
        "fields": fields,
        "action": decision.action,
        "decision": decision.action,
        "status": "applied" if applied_target_uri else "skipped",
        "applied": bool(applied_target_uri),
        "failed": False,
        "skipped": not bool(applied_target_uri),
        "reason": decision.reason,
        "confidence": decision.confidence,
        "candidate_count": len(candidates),
        "candidate_uris": list(decision.candidate_uris),
        "scope": {
            "memory_type": scope.memory_type,
            "operation_uri": scope.operation_uri,
            "parent_uri": scope.parent_uri,
            "lock_uri": scope.lock_uri,
            "candidate_count": len(scope.candidate_uris),
        },
        "target_uri": decision.target_uri,
        "applied_to_uri": applied_target_uri,
        "redirected": bool(applied_target_uri and original_uris != [applied_target_uri]),
        "adapter_telemetry": dict(decision.telemetry or {}),
    }


def _rewrite_resolved_links(
    operations: ResolvedOperations,
    from_uris: Iterable[str],
    target_uri: str,
) -> None:
    source_uris = {uri for uri in from_uris if uri and uri != target_uri}
    if not source_uris:
        return
    for link in operations.resolved_links or []:
        if link.from_uri in source_uris:
            link.from_uri = target_uri
        if link.to_uri in source_uris:
            link.to_uri = target_uri


async def _acquire_admission_scope_lock(
    *,
    scope: AdmissionScope,
    provider: Any,
    ctx: RequestContext,
    viking_fs: VikingFS,
    require_lock: bool,
) -> Optional[str]:
    if not require_lock:
        return None
    if not scope.lock_uri or viking_fs is None:
        raise RuntimeError("admission scope lock requires lock_uri and viking_fs")

    try:
        from openviking.storage.transaction import get_lock_manager

        lock_manager = get_lock_manager()
        lock_path = viking_fs._uri_to_path(scope.lock_uri, ctx=ctx)
    except Exception as exc:
        get_current_telemetry().increment("memory.admission.lock.unavailable")
        raise RuntimeError(f"admission scope lock unavailable: {exc}") from exc

    handle = getattr(provider, "_transaction_handle", None)
    if handle is None:
        raise RuntimeError("admission scope lock requires an active transaction handle")

    if lock_path in list(getattr(handle, "locks", []) or []):
        return lock_path
    acquired = await lock_manager.acquire_exact_path_batch(handle, [lock_path], timeout=None)
    if not acquired:
        raise RuntimeError(f"failed to acquire admission scope lock: {scope.lock_uri}")
    return lock_path


async def apply_admission_adapters(
    *,
    operations: ResolvedOperations,
    adapters: list[OperationAdmissionAdapter],
    registry: Any,
    provider: Any,
    ctx: RequestContext,
    viking_fs: VikingFS,
    require_lock: bool = False,
) -> list[AdmissionDecision]:
    """Run admission adapters in-place over extracted upsert operations."""

    decisions: list[AdmissionDecision] = []
    telemetry = get_current_telemetry()

    for operation in operations.upsert_operations or []:
        schema = _get_schema(registry, operation.memory_type)
        for adapter in adapters:
            if not adapter.supports(operation, schema, ctx):
                continue
            scope = await adapter.derive_scope(operation, schema, provider, ctx, viking_fs)
            await _acquire_admission_scope_lock(
                scope=scope,
                provider=provider,
                ctx=ctx,
                viking_fs=viking_fs,
                require_lock=require_lock,
            )
            candidates = await adapter.refresh_candidates(
                scope,
                ctx,
                viking_fs,
                provider=provider,
            )
            decision = await adapter.decide(operation, candidates, scope)
            original_uris = list(operation.uris)
            input_shape = "edit" if operation.old_memory_file_content is not None else "create"
            fields = _operation_field_traces(operation, schema)
            old_uris, target_uri = adapter.apply_decision(operation, decision)
            if target_uri:
                _rewrite_resolved_links(operations, old_uris, target_uri)
            decision.trace = _build_admission_trace(
                operation=operation,
                scope=scope,
                candidates=candidates,
                decision=decision,
                original_uris=original_uris,
                input_shape=input_shape,
                fields=fields,
                output_uris=list(operation.uris),
                applied_target_uri=target_uri,
            )
            decisions.append(decision)
            telemetry.increment(f"memory.admission.{decision.action}")
            tracer.info(
                "memory admission decision: "
                f"memory_type={operation.memory_type} "
                f"action={decision.action} "
                f"reason={decision.reason} "
                f"confidence={decision.confidence} "
                f"target_uri={decision.target_uri} "
                f"candidates={decision.candidate_uris}"
            )
            break

    return decisions
