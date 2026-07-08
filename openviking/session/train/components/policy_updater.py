# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""PolicyUpdater component implementations."""

from __future__ import annotations

import re
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from openviking.session.memory.dataclass import (
    MemoryFile,
    ResolvedOperation,
    ResolvedOperations,
    StoredLink,
)
from openviking.session.memory.memory_type_registry import create_default_registry
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.session.train.domain import (
    Policy,
    PolicyApplyResult,
    PolicyPlanItem,
    PolicySet,
    PolicyUpdatePlan,
)
from openviking.storage.transaction import LockContext, get_lock_manager
from openviking.storage.viking_fs import get_viking_fs
from openviking.telemetry import tracer

_EXPERIENCE_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(slots=True)
class DryRunPolicyUpdater:
    """PolicyUpdater that records a plan without writing files.

    Unlike a pure no-op, this updater simulates executable plan items into an
    updated ExperienceSet snapshot, which makes tests and offline review useful
    before enabling a writing updater.
    """

    simulate: bool = True

    @tracer("train.policy_updater.dry_run.apply", ignore_result=True, ignore_args=True)
    async def apply(
        self,
        plan: PolicyUpdatePlan,
        policy_set: PolicySet,
        context: Any = None,
        *,
        transaction_handle: Any = None,
    ) -> PolicyApplyResult:
        del transaction_handle
        del context
        updated_policy_set = (
            _apply_items_to_snapshot(plan.items, policy_set)
            if self.simulate and plan.items
            else policy_set
        )
        return PolicyApplyResult(
            updated_policy_set=updated_policy_set,
            written_uris=[],
            metadata={
                "dry_run": True,
                "simulated": self.simulate,
                "plan": plan.metadata,
                "item_count": len(plan.items),
            },
        )


@dataclass(slots=True)
class MemoryFilePolicyUpdater:
    """PolicyUpdater that writes policy files via VikingFS.

    It consumes executable ``upsert`` and ``delete`` plan items. The updater
    performs a lightweight base-content guard when ``before_content`` is
    available to avoid blindly overwriting or deleting a diverged policy set
    snapshot.
    """

    viking_fs: Any = None
    vikingdb: Any = None
    exact_file_lock: bool = False

    @property
    def plan_outside_policy_lock(self) -> bool:
        """Whether the engine may plan outside the policy root tree lock.

        This is only safe when apply re-reads the latest target files under
        exact file locks and refuses stale operations.
        """

        return self.exact_file_lock

    @tracer("train.policy_updater.memory_file.apply", ignore_result=True, ignore_args=True)
    async def apply(
        self,
        plan: PolicyUpdatePlan,
        policy_set: PolicySet,
        context: Any = None,
        *,
        transaction_handle: Any = None,
    ) -> PolicyApplyResult:
        viking_fs = self.viking_fs or get_viking_fs()
        if viking_fs is None:
            raise RuntimeError("VikingFS is required to apply policy update plans")

        active_handle = transaction_handle
        apply_traces: list[dict[str, Any]] = []
        preflight_errors: list[str] = []
        safe_plan = plan
        async with _policy_exact_lock(
            viking_fs,
            _plan_target_uris(plan, policy_set.root_uri),
            ctx=context,
            enabled=self.exact_file_lock and transaction_handle is None,
        ) as exact_handle:
            active_handle = exact_handle or transaction_handle
            if self.exact_file_lock:
                safe_plan, preflight_errors, apply_traces = await _preflight_exact_plan(
                    plan=plan,
                    policy_set=policy_set,
                    viking_fs=viking_fs,
                    context=context,
                )
            if preflight_errors:
                return PolicyApplyResult(
                    updated_policy_set=policy_set,
                    written_uris=[],
                    deleted_uris=[],
                    errors=preflight_errors,
                    metadata=_apply_metadata(
                        plan=plan,
                        operations=None,
                        apply_traces=apply_traces,
                    ),
                )

            updated_policy_set = _apply_items_to_snapshot(safe_plan.items, policy_set)
            operations, operation_preflight_errors = _plan_to_resolved_operations(
                plan=safe_plan,
                policy_set=policy_set,
                updated_policy_set=updated_policy_set,
            )
            if operation_preflight_errors:
                _mark_preflight_ok_traces(apply_traces, status="failed")
                return PolicyApplyResult(
                    updated_policy_set=policy_set,
                    written_uris=[],
                    deleted_uris=[],
                    errors=operation_preflight_errors,
                    metadata=_apply_metadata(
                        plan=plan,
                        operations=operations,
                        apply_traces=apply_traces,
                    ),
                )

            updater = MemoryUpdater(
                registry=create_default_registry(),
                vikingdb=self.vikingdb,
                transaction_handle=active_handle,
            )
            updater._viking_fs = viking_fs

            apply_result = await updater.apply_operations(
                operations,
                context,
                extract_context=None,
                isolation_handler=None,
            )
        errors = [f"{uri}: {exc}" for uri, exc in apply_result.errors]
        _mark_preflight_ok_traces(
            apply_traces,
            status="failed" if errors else "applied",
        )
        result_policy_set = updated_policy_set if not errors else policy_set
        if self.exact_file_lock and not errors:
            result_policy_set = await _reload_policy_set_if_possible(result_policy_set)

        return PolicyApplyResult(
            updated_policy_set=result_policy_set,
            written_uris=list(apply_result.written_uris + apply_result.edited_uris),
            deleted_uris=list(apply_result.deleted_uris),
            errors=errors,
            metadata=_apply_metadata(
                plan=plan,
                operations=operations,
                apply_traces=apply_traces,
            ),
        )


def _apply_items_to_snapshot(
    items: list[PolicyPlanItem], policy_set: PolicySet
) -> PolicySet:
    policies_by_uri = {policy.uri: policy for policy in policy_set.policies}
    result = list(policy_set.policies)

    for item in items:
        uri = _target_uri(item, policy_set.root_uri)

        if item.kind == "delete":
            existing = policies_by_uri.get(uri) or _find_policy(
                PolicySet(
                    policy_set.root_uri,
                    result,
                    metadata=dict(policy_set.metadata),
                    viking_fs=policy_set.viking_fs,
                    request_context=policy_set.request_context,
                ),
                uri=None,
                name=item.target_name,
            )
            remove_uri = existing.uri if existing is not None else uri
            result = [
                policy
                for policy in result
                if policy.uri != remove_uri and policy.name != item.target_name
            ]
            policies_by_uri.pop(remove_uri, None)
            policies_by_uri.pop(uri, None)
            continue

        if item.kind != "upsert" or item.after_content is None:
            continue
        existing = policies_by_uri.get(uri) or _find_policy(
            PolicySet(
                policy_set.root_uri,
                result,
                metadata=dict(policy_set.metadata),
                viking_fs=policy_set.viking_fs,
                request_context=policy_set.request_context,
            ),
            uri=None,
            name=item.target_name,
        )
        metadata = dict(existing.metadata) if existing is not None else {}
        metadata.update(item.metadata.get("patch_metadata", {}))
        metadata.setdefault("memory_type", item.memory_type or "experiences")
        metadata["experience_name"] = item.target_name
        version = (existing.version + 1) if existing is not None else 1
        updated = Policy(
            name=item.target_name,
            uri=uri,
            version=version,
            status=(existing.status if existing is not None else "draft"),
            content=item.after_content,
            metadata=metadata,
            links=list(existing.links or []) if existing is not None else [],
            backlinks=list(existing.backlinks or []) if existing is not None else [],
        )
        if existing is None:
            result.append(updated)
        else:
            result = [updated if policy.uri == existing.uri else policy for policy in result]
        policies_by_uri[uri] = updated

    result.sort(key=lambda policy: policy.uri)
    return PolicySet(
        root_uri=policy_set.root_uri,
        policies=result,
        metadata=dict(policy_set.metadata),
        viking_fs=policy_set.viking_fs,
        request_context=policy_set.request_context,
    )


def _find_policy(
    policy_set: PolicySet,
    *,
    uri: str | None,
    name: str,
) -> Policy | None:
    for policy in policy_set.policies:
        if uri and policy.uri == uri:
            return policy
        if not uri and policy.name == name:
            return policy
    return None


def _target_uri(item: PolicyPlanItem, root_uri: str) -> str:
    if item.target_uri:
        return item.target_uri
    return f"{root_uri.rstrip('/')}/{_safe_experience_filename(item.target_name)}.md"


def _plan_target_uris(plan: PolicyUpdatePlan, root_uri: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in plan.items:
        uri = _target_uri(item, root_uri)
        if uri in seen:
            continue
        seen.add(uri)
        result.append(uri)
    return result


@asynccontextmanager
async def _policy_exact_lock(
    viking_fs: Any,
    uris: list[str],
    *,
    ctx: Any,
    enabled: bool,
) -> AsyncIterator[Any | None]:
    if not enabled or not uris:
        yield None
        return

    uri_to_path = getattr(viking_fs, "_uri_to_path", None)
    if uri_to_path is None:
        raise RuntimeError("VikingFS must provide _uri_to_path for exact file locking")

    paths = [uri_to_path(uri, ctx=ctx) for uri in uris]
    async with LockContext(get_lock_manager(), paths, lock_mode="exact") as handle:
        yield handle


async def _preflight_exact_plan(
    *,
    plan: PolicyUpdatePlan,
    policy_set: PolicySet,
    viking_fs: Any,
    context: Any,
) -> tuple[PolicyUpdatePlan, list[str], list[dict[str, Any]]]:
    safe_items: list[PolicyPlanItem] = []
    errors: list[str] = []
    traces: list[dict[str, Any]] = []

    for item in plan.items:
        uri = _target_uri(item, policy_set.root_uri)
        current, read_error = await _read_current_policy_file(viking_fs, uri, context)
        trace = _new_apply_trace(item=item, uri=uri)
        if read_error is not None:
            trace.update(
                {
                    "status": "failed_read_latest",
                    "stale_detected": True,
                    "error": str(read_error),
                }
            )
            errors.append(f"{uri}: failed to read latest policy file: {read_error}")
            traces.append(trace)
            continue

        before = _normalize_optional_guard_content(item.before_content)
        current_content = (
            _normalize_guard_content(current.plain_content())
            if current is not None
            else None
        )

        if item.kind == "upsert":
            if current is None and before is not None:
                trace.update(
                    {
                        "status": "skipped_stale_deleted",
                        "stale_detected": True,
                    }
                )
                traces.append(trace)
                continue
            if current is not None and before is None:
                trace.update(
                    {
                        "status": "skipped_stale_unread_existing",
                        "stale_detected": True,
                    }
                )
                traces.append(trace)
                continue
            if current_content is not None and before is not None and current_content != before:
                trace.update(
                    {
                        "status": "failed_base_content_mismatch",
                        "stale_detected": True,
                        "rewrite_attempted": False,
                    }
                )
                errors.append(
                    "base content mismatch for "
                    f"{item.target_name}: expected gradient before_content"
                )
                traces.append(trace)
                continue
            trace["status"] = "preflight_ok"
            safe_items.append(item)
            traces.append(trace)
            continue

        if item.kind == "delete":
            if current is None:
                trace.update(
                    {
                        "status": "skipped_stale_deleted",
                        "stale_detected": True,
                    }
                )
                traces.append(trace)
                continue
            if before is not None and current_content != before:
                trace.update(
                    {
                        "status": "failed_base_content_mismatch",
                        "stale_detected": True,
                        "rewrite_attempted": False,
                    }
                )
                errors.append(
                    "base content mismatch for "
                    f"{item.target_name}: expected gradient before_content"
                )
                traces.append(trace)
                continue
            trace["status"] = "preflight_ok"
            safe_items.append(item)
            traces.append(trace)
            continue

        trace.update({"status": "skipped_unsupported_item_kind"})
        traces.append(trace)

    return PolicyUpdatePlan(items=safe_items, metadata=dict(plan.metadata)), errors, traces


async def _read_current_policy_file(
    viking_fs: Any,
    uri: str,
    context: Any,
) -> tuple[MemoryFile | None, Exception | None]:
    try:
        content = await viking_fs.read_file(uri, ctx=context)
    except Exception as exc:  # keep storage adapters decoupled from CLI exceptions
        if _is_not_found_error(exc):
            return None, None
        return None, exc
    if not content:
        return None, None
    try:
        return MemoryFileUtils.read(content, uri=uri), None
    except Exception as exc:
        return None, exc


def _is_not_found_error(exc: Exception) -> bool:
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return True
    name = exc.__class__.__name__.lower()
    return "notfound" in name or "not_found" in name


def _new_apply_trace(*, item: PolicyPlanItem, uri: str) -> dict[str, Any]:
    return {
        "uri": uri,
        "memory_type": item.memory_type or "experiences",
        "target_name": item.target_name,
        "operation": item.kind,
        "field": "content",
        "merge_op": "replace",
        "input_shape": "PolicyPlanItem",
        "wrapper_shape": "full_file",
        "stale_detected": False,
        "rewrite_attempted": False,
        "status": "pending",
    }


def _mark_preflight_ok_traces(traces: list[dict[str, Any]], *, status: str) -> None:
    for trace in traces:
        if trace.get("status") == "preflight_ok":
            trace["status"] = status


def _apply_metadata(
    *,
    plan: PolicyUpdatePlan,
    operations: ResolvedOperations | None,
    apply_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dry_run": False,
        "item_count": len(plan.items),
        "operation_upsert_count": len(operations.upsert_operations) if operations else 0,
        "operation_delete_count": len(operations.delete_file_contents) if operations else 0,
    }
    if apply_traces:
        metadata["apply_trace"] = apply_traces
        metadata["apply_trace_summary"] = dict(
            Counter(str(trace.get("status", "unknown")) for trace in apply_traces)
        )
    return metadata


async def _reload_policy_set_if_possible(policy_set: PolicySet) -> PolicySet:
    if policy_set.viking_fs is None or policy_set.request_context is None:
        return policy_set
    return await policy_set.reload()



def _plan_to_resolved_operations(
    *,
    plan: PolicyUpdatePlan,
    policy_set: PolicySet,
    updated_policy_set: PolicySet,
) -> tuple[ResolvedOperations, list[str]]:
    upserts: list[ResolvedOperation] = []
    deletes: list[MemoryFile] = []
    links: list[StoredLink] = []
    errors: list[str] = []

    for item in plan.items:
        uri = _target_uri(item, policy_set.root_uri)
        current = _find_policy(policy_set, uri=uri, name=item.target_name)
        if (
            current is not None
            and item.before_content is not None
            and _normalize_guard_content(current.content)
            != _normalize_guard_content(item.before_content)
        ):
            errors.append(
                "base content mismatch for "
                f"{item.target_name}: expected gradient before_content"
            )
            continue

        if item.kind == "delete":
            deletes.append(_policy_or_plan_item_memory_file(item, uri=uri, current=current))
            continue

        if item.kind != "upsert":
            continue
        if item.after_content is None:
            errors.append(f"missing after_content for {item.target_name}")
            continue

        updated = _find_policy(updated_policy_set, uri=uri, name=item.target_name)
        if updated is None:
            errors.append(
                f"planned policy not found after simulation: {item.target_name}"
            )
            continue

        upserts.append(
            ResolvedOperation(
                old_memory_file_content=_policy_to_memory_file(current)
                if current is not None
                else None,
                memory_fields={
                    **dict(updated.metadata),
                    "memory_type": item.memory_type or "experiences",
                    "experience_name": updated.name,
                    "content": updated.content,
                    "status": updated.status,
                },
                memory_type=item.memory_type or "experiences",
                uris=[uri],
            )
        )
        links.extend(_source_trajectory_links(exp_uri=uri, links=item.links))

    return (
        ResolvedOperations(
            upsert_operations=upserts,
            delete_file_contents=deletes,
            errors=[],
            resolved_links=links,
        ),
        errors,
    )


def _policy_or_plan_item_memory_file(
    item: PolicyPlanItem,
    *,
    uri: str,
    current: Policy | None,
) -> MemoryFile:
    if current is not None:
        return _policy_to_memory_file(current)
    return MemoryFile(
        uri=uri,
        content=item.before_content or "",
        memory_type=item.memory_type or "experiences",
        extra_fields={
            "memory_type": item.memory_type or "experiences",
            "experience_name": item.target_name,
            **({"version": item.base_version} if item.base_version is not None else {}),
        },
    )


def _policy_to_memory_file(policy: Policy | None) -> MemoryFile | None:
    if policy is None:
        return None
    return MemoryFile(
        uri=policy.uri,
        content=policy.content,
        links=list(policy.links or []),
        backlinks=list(policy.backlinks or []),
        memory_type="experiences",
        extra_fields={
            **dict(policy.metadata),
            "memory_type": "experiences",
            "experience_name": policy.name,
            "version": policy.version,
            "status": policy.status,
        },
    )


def _source_trajectory_links(
    *,
    exp_uri: str,
    links: list[StoredLink],
) -> list[StoredLink]:
    result: list[StoredLink] = []
    seen: set[tuple[str, str | None]] = set()
    for link in links or []:
        if (
            link.link_type != "derived_from"
            or not link.to_uri
            or "/memories/trajectories/" not in link.to_uri
        ):
            continue
        key = (link.to_uri, link.match_text)
        if key in seen:
            continue
        seen.add(key)
        update = {"from_uri": exp_uri, "match_text": None, "description": ""}
        if not link.created_at:
            update["created_at"] = datetime.now(timezone.utc).isoformat()
        result.append(link.model_copy(update=update))
    return result


def _safe_experience_filename(name: str) -> str:
    filename = _EXPERIENCE_NAME_RE.sub("_", name.strip()).strip("._-")
    return filename or "new_experience"


def _normalize_guard_content(content: str) -> str:
    return content.strip()


def _normalize_optional_guard_content(content: str | None) -> str | None:
    if content is None:
        return None
    return _normalize_guard_content(content)
