# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Admission adapter for agent experience create proposals."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Optional

from openviking.server.identity import RequestContext
from openviking.session.memory.admission import (
    AdmissionDecision,
    AdmissionScope,
    OperationAdmissionAdapter,
)
from openviking.session.memory.dataclass import MemoryFile, ResolvedOperation
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.storage.viking_fs import VikingFS

EXPERIENCE_MEMORY_TYPE = "experiences"
_ADMISSION_LOCK_FILENAME = ".experience_admission.ovlock"
_COMPARATIVE_INSIGHT_MIN_CONFIDENCE = 0.55
_COMPARATIVE_INSIGHT_MIN_SECTION_SCORE = 0.42
_TEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _name_tokens(value: str) -> set[str]:
    return {token for token in _normalize_name(value).split("_") if token}


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = _name_tokens(left)
    right_tokens = _name_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _text_tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    return {token for token in tokens if len(token) > 2 and token not in _TEXT_STOPWORDS}


def _jaccard(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _content_sections(content: Any) -> dict[str, str]:
    text = str(content or "")
    matches = list(re.finditer(r"^##\s+([^\n#]+?)\s*$", text, flags=re.MULTILINE))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = _normalize_name(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[heading] = text[start:end].strip()
    return sections


def _comparative_insight_scores(left_content: Any, right_content: Any) -> dict[str, float]:
    left_sections = _content_sections(left_content)
    right_sections = _content_sections(right_content)
    situation = _jaccard(
        _text_tokens(left_sections.get("situation")),
        _text_tokens(right_sections.get("situation")),
    )
    approach = _jaccard(
        _text_tokens(left_sections.get("approach")),
        _text_tokens(right_sections.get("approach")),
    )
    reflect = _jaccard(
        _text_tokens(left_sections.get("reflect")),
        _text_tokens(right_sections.get("reflect")),
    )
    confidence = (0.35 * situation) + (0.2 * approach) + (0.45 * reflect)
    return {
        "situation_score": round(situation, 4),
        "approach_score": round(approach, 4),
        "reflect_score": round(reflect, 4),
        "comparative_insight_confidence": round(confidence, 4),
    }


def _uri_parent(uri: str) -> str:
    return uri.rsplit("/", 1)[0] if "/" in uri else ""


def _uri_stem(uri: str) -> str:
    stem = PurePosixPath(uri).name
    if stem.endswith(".md"):
        stem = stem[:-3]
    return stem


def _is_experience_file_uri(uri: str) -> bool:
    if not uri.endswith(".md"):
        return False
    if uri.endswith("/.overview.md") or uri.endswith("/.abstract.md"):
        return False
    return True


def _is_same_parent_experience_uri(uri: str, parent_uri: Optional[str]) -> bool:
    return bool(parent_uri and _uri_parent(uri) == parent_uri and _is_experience_file_uri(uri))


def _candidate_experience_name(memory_file: MemoryFile) -> str:
    value = memory_file.extra_fields.get("experience_name")
    if value:
        return str(value)
    return _uri_stem(memory_file.uri)


class AgentExperienceAdmissionAdapter(OperationAdmissionAdapter):
    """Redirect high-confidence duplicate experience creates to updates."""

    def __init__(self, mode: str = "name_only") -> None:
        if mode not in {"name_only", "comparative_insight"}:
            raise ValueError(
                "AgentExperienceAdmissionAdapter mode must be 'name_only' "
                "or 'comparative_insight'"
            )
        self.mode = mode

    def supports(self, operation: ResolvedOperation, schema: Any, ctx: RequestContext) -> bool:
        if operation.memory_type != EXPERIENCE_MEMORY_TYPE:
            return False
        if operation.old_memory_file_content is not None:
            return False
        if operation.memory_fields.get("supersedes"):
            return False
        return bool(operation.uris)

    async def derive_scope(
        self,
        operation: ResolvedOperation,
        schema: Any,
        provider: Any,
        ctx: RequestContext,
        viking_fs: VikingFS,
    ) -> AdmissionScope:
        operation_uri = operation.uris[0] if operation.uris else None
        parent_uri = _uri_parent(operation_uri or "")
        prefetched_uris = [
            uri
            for uri in getattr(provider, "prefetched_uris", []) or []
            if _is_same_parent_experience_uri(uri, parent_uri)
        ]
        read_uris = [
            uri
            for uri in getattr(provider, "read_file_contents", {}) or {}
            if _is_same_parent_experience_uri(uri, parent_uri)
        ]
        candidate_uris = list(dict.fromkeys(prefetched_uris + read_uris))
        return AdmissionScope(
            memory_type=EXPERIENCE_MEMORY_TYPE,
            operation_uri=operation_uri,
            parent_uri=parent_uri,
            lock_uri=f"{parent_uri}/{_ADMISSION_LOCK_FILENAME}" if parent_uri else None,
            candidate_uris=candidate_uris,
        )

    async def refresh_candidates(
        self,
        scope: AdmissionScope,
        ctx: RequestContext,
        viking_fs: VikingFS,
        provider: Any = None,
    ) -> list[MemoryFile]:
        read_file_contents = getattr(provider, "read_file_contents", {}) or {}
        candidate_uris = list(scope.candidate_uris)
        if viking_fs is not None and scope.parent_uri:
            try:
                entries = await viking_fs.ls(scope.parent_uri, output="original", ctx=ctx)
                for entry in entries or []:
                    uri = str(entry.get("uri", "")) if isinstance(entry, dict) else ""
                    if not uri:
                        continue
                    if _is_same_parent_experience_uri(uri, scope.parent_uri):
                        candidate_uris.append(uri)
            except Exception:
                pass
        candidate_uris = [
            uri
            for uri in dict.fromkeys(candidate_uris)
            if _is_same_parent_experience_uri(uri, scope.parent_uri)
        ]

        candidates: list[MemoryFile] = []
        for uri in candidate_uris:
            if uri == scope.operation_uri:
                continue
            memory_file = read_file_contents.get(uri)
            if memory_file is None and provider is not None:
                read_file = getattr(provider, "read_file", None)
                if callable(read_file):
                    await read_file(uri)
                    memory_file = read_file_contents.get(uri)
            if memory_file is None and viking_fs is not None:
                try:
                    raw = await viking_fs.read_file(uri, ctx=ctx)
                    if raw:
                        memory_file = MemoryFileUtils.read(raw, uri=uri)
                except Exception:
                    memory_file = None
            if memory_file is None:
                continue
            if memory_file.memory_type and memory_file.memory_type != EXPERIENCE_MEMORY_TYPE:
                continue
            candidates.append(memory_file)
        return candidates

    async def decide(
        self,
        operation: ResolvedOperation,
        candidates: list[MemoryFile],
        scope: AdmissionScope,
    ) -> AdmissionDecision:
        proposed_name = str(operation.memory_fields.get("experience_name") or "")
        proposed_normalized = _normalize_name(proposed_name)
        operation_stem = _normalize_name(_uri_stem(scope.operation_uri or ""))
        candidate_uris = [candidate.uri for candidate in candidates]

        for candidate in candidates:
            candidate_name = _candidate_experience_name(candidate)
            candidate_normalized = _normalize_name(candidate_name)
            candidate_stem = _normalize_name(_uri_stem(candidate.uri))
            if proposed_normalized and proposed_normalized in {
                candidate_normalized,
                candidate_stem,
            }:
                return AdmissionDecision(
                    action="redirect_update",
                    target_uri=candidate.uri,
                    target_memory_file=candidate,
                    reason="same_experience_name",
                    confidence=1.0,
                    candidate_uris=candidate_uris,
                )
            if operation_stem and operation_stem in {candidate_normalized, candidate_stem}:
                return AdmissionDecision(
                    action="redirect_update",
                    target_uri=candidate.uri,
                    target_memory_file=candidate,
                    reason="same_uri_stem",
                    confidence=1.0,
                    candidate_uris=candidate_uris,
                )

        for candidate in candidates:
            candidate_name = _candidate_experience_name(candidate)
            score = _token_jaccard(proposed_name, candidate_name)
            if score >= 0.8:
                return AdmissionDecision(
                    action="redirect_update",
                    target_uri=candidate.uri,
                    target_memory_file=candidate,
                    reason="near_experience_name",
                    confidence=score,
                    candidate_uris=candidate_uris,
                    telemetry={"name_jaccard": score},
                )

        if self.mode == "comparative_insight":
            proposed_content = operation.memory_fields.get("content")
            for candidate in candidates:
                scores = _comparative_insight_scores(proposed_content, candidate.content)
                confidence = scores["comparative_insight_confidence"]
                if (
                    confidence >= _COMPARATIVE_INSIGHT_MIN_CONFIDENCE
                    and scores["situation_score"] >= _COMPARATIVE_INSIGHT_MIN_SECTION_SCORE
                    and scores["reflect_score"] >= _COMPARATIVE_INSIGHT_MIN_SECTION_SCORE
                ):
                    return AdmissionDecision(
                        action="redirect_update",
                        target_uri=candidate.uri,
                        target_memory_file=candidate,
                        reason="same_comparative_insight_boundary",
                        confidence=confidence,
                        candidate_uris=candidate_uris,
                        telemetry=scores,
                    )

        return AdmissionDecision(
            action="allow_with_telemetry" if candidates else "allow_create",
            reason="no_high_confidence_candidate",
            confidence=0.0,
            candidate_uris=candidate_uris,
        )

    def apply_decision(
        self,
        operation: ResolvedOperation,
        decision: AdmissionDecision,
    ) -> tuple[list[str], Optional[str]]:
        if decision.action not in {"redirect_update", "replacement"}:
            return [], None
        if not decision.target_uri or decision.target_memory_file is None:
            return [], None

        old_uris = list(operation.uris)
        operation.uris = [decision.target_uri]
        operation.old_memory_file_content = decision.target_memory_file

        old_name = decision.target_memory_file.extra_fields.get("experience_name")
        if old_name:
            operation.memory_fields["experience_name"] = old_name

        return old_uris, decision.target_uri
