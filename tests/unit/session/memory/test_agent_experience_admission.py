# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
from types import SimpleNamespace

from openviking.session.memory.admission import apply_admission_adapters
from openviking.session.memory.agent_experience_admission import AgentExperienceAdmissionAdapter
from openviking.session.memory.dataclass import (
    MemoryField,
    MemoryFile,
    MemoryTypeSchema,
    ResolvedOperation,
    ResolvedOperations,
    StoredLink,
)
from openviking.session.memory.merge_op import FieldType, MergeOp
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.telemetry import OperationTelemetry, bind_telemetry


def _experience(uri: str, name: str) -> MemoryFile:
    return MemoryFile(
        uri=uri,
        content="## Situation\n- old\n\n## Approach\n- old\n\n## Reflect\n- old",
        memory_type="experiences",
        extra_fields={"experience_name": name},
    )


def _operation(uri: str, name: str, **fields) -> ResolvedOperation:
    memory_fields = {
        "experience_name": name,
        "content": "## Situation\n- new\n\n## Approach\n- new\n\n## Reflect\n- new",
    }
    memory_fields.update(fields)
    return ResolvedOperation(
        old_memory_file_content=None,
        memory_fields=memory_fields,
        memory_type="experiences",
        uris=[uri],
    )


def test_agent_experience_admission_redirects_same_name_create_to_update():
    async def run():
        old_uri = "viking://agent/a/memories/experiences/booking_duplicate_handling.md"
        new_uri = "viking://agent/a/memories/experiences/duplicate_booking_handling.md"
        old_memory = _experience(old_uri, "booking_duplicate_handling")
        op = _operation(new_uri, "duplicate_booking_handling")
        operations = ResolvedOperations(
            upsert_operations=[op],
            delete_file_contents=[],
            errors=[],
            resolved_links=[
                StoredLink(
                    from_uri=new_uri,
                    to_uri="viking://agent/a/memories/trajectories/20260602000000000000.md",
                    link_type="derived_from",
                )
            ],
        )
        provider = SimpleNamespace(
            prefetched_uris=[old_uri],
            read_file_contents={old_uri: old_memory},
            _transaction_handle=None,
        )

        telemetry = OperationTelemetry(operation="session.commit", enabled=True)
        with bind_telemetry(telemetry):
            decisions = await apply_admission_adapters(
                operations=operations,
                adapters=[AgentExperienceAdmissionAdapter()],
                registry={
                    "experiences": MemoryTypeSchema(
                        memory_type="experiences",
                        fields=[
                            MemoryField(
                                name="content",
                                field_type=FieldType.STRING,
                                merge_op=MergeOp.REPLACE,
                            ),
                            MemoryField(
                                name="experience_name",
                                field_type=FieldType.STRING,
                                merge_op=MergeOp.IMMUTABLE,
                            ),
                        ],
                    )
                },
                provider=provider,
                ctx=None,
                viking_fs=None,
                require_lock=False,
            )
        admission_summary = telemetry.finish().summary["memory"]["admission"]["trace"]

        assert decisions[0].action == "redirect_update"
        assert decisions[0].reason == "near_experience_name"
        assert op.uris == [old_uri]
        assert op.old_memory_file_content is old_memory
        assert op.memory_fields["experience_name"] == "booking_duplicate_handling"
        assert operations.resolved_links[0].from_uri == old_uri
        assert decisions[0].trace["uri"] == new_uri
        assert decisions[0].trace["output_uris"] == [old_uri]
        assert decisions[0].trace["memory_type"] == "experiences"
        assert decisions[0].trace["trace_type"] == "admission_decision"
        assert decisions[0].trace["input_shape"] == "create"
        assert decisions[0].trace["action"] == "redirect_update"
        assert decisions[0].trace["decision"] == "redirect_update"
        assert decisions[0].trace["status"] == "applied"
        assert decisions[0].trace["applied"] is True
        assert decisions[0].trace["failed"] is False
        assert decisions[0].trace["skipped"] is False
        assert decisions[0].trace["reason"] == "near_experience_name"
        assert decisions[0].trace["candidate_count"] == 1
        assert decisions[0].trace["applied_to_uri"] == old_uri
        assert decisions[0].trace["redirected"] is True
        assert {
            (
                field["field"],
                field["merge_op"],
                field["input_shape"],
                field["wrapper_shape"],
            )
            for field in decisions[0].trace["fields"]
        } == {
            ("content", "replace", "str", None),
            ("experience_name", "immutable", "str", None),
        }
        assert admission_summary["total"] == 1
        assert admission_summary["action"] == {"redirect_update": 1}
        assert admission_summary["status"] == {"applied": 1}
        assert admission_summary["reason"] == {"near_experience_name": 1}
        assert admission_summary["redirected"] == 1
        assert admission_summary["candidates_total"] == 1

    asyncio.run(run())


def test_agent_experience_admission_allows_uncertain_create_with_telemetry():
    async def run():
        old_uri = "viking://agent/a/memories/experiences/cancel_order_flow.md"
        new_uri = "viking://agent/a/memories/experiences/add_baggage_flow.md"
        old_memory = _experience(old_uri, "cancel_order_flow")
        op = _operation(new_uri, "add_baggage_flow")
        provider = SimpleNamespace(
            prefetched_uris=[old_uri],
            read_file_contents={old_uri: old_memory},
            _transaction_handle=None,
        )

        decisions = await apply_admission_adapters(
            operations=ResolvedOperations(
                upsert_operations=[op],
                delete_file_contents=[],
                errors=[],
            ),
            adapters=[AgentExperienceAdmissionAdapter()],
            registry={},
            provider=provider,
            ctx=None,
            viking_fs=None,
            require_lock=False,
        )

        assert decisions[0].action == "allow_with_telemetry"
        assert op.uris == [new_uri]
        assert op.old_memory_file_content is None
        assert decisions[0].trace["status"] == "skipped"
        assert decisions[0].trace["applied"] is False
        assert decisions[0].trace["failed"] is False
        assert decisions[0].trace["skipped"] is True
        assert decisions[0].trace["applied_to_uri"] is None
        assert decisions[0].trace["redirected"] is False
        assert decisions[0].trace["candidate_count"] == 1

    asyncio.run(run())


def test_agent_experience_admission_dedupes_links_after_redirect():
    async def run():
        old_uri = "viking://agent/a/memories/experiences/booking_duplicate_handling.md"
        new_uri = "viking://agent/a/memories/experiences/duplicate_booking_handling.md"
        trajectory_uri = "viking://agent/a/memories/trajectories/20260602000000000000.md"
        old_memory = _experience(old_uri, "booking_duplicate_handling")
        op = _operation(new_uri, "duplicate_booking_handling")
        operations = ResolvedOperations(
            upsert_operations=[op],
            delete_file_contents=[],
            errors=[],
            resolved_links=[
                StoredLink(from_uri=old_uri, to_uri=trajectory_uri, link_type="derived_from"),
                StoredLink(from_uri=new_uri, to_uri=trajectory_uri, link_type="derived_from"),
            ],
        )
        provider = SimpleNamespace(
            prefetched_uris=[old_uri],
            read_file_contents={old_uri: old_memory},
            _transaction_handle=None,
        )

        decisions = await apply_admission_adapters(
            operations=operations,
            adapters=[AgentExperienceAdmissionAdapter()],
            registry={},
            provider=provider,
            ctx=None,
            viking_fs=None,
            require_lock=False,
        )

        assert decisions[0].action == "redirect_update"
        assert [(link.from_uri, link.to_uri, link.link_type) for link in operations.resolved_links] == [
            (old_uri, trajectory_uri, "derived_from")
        ]

    asyncio.run(run())


def test_agent_experience_admission_refreshes_directory_candidates_under_lock():
    async def run():
        old_uri = "viking://agent/a/memories/experiences/booking_duplicate_handling.md"
        new_uri = "viking://agent/a/memories/experiences/duplicate_booking_handling.md"
        old_memory = _experience(old_uri, "booking_duplicate_handling")
        op = _operation(new_uri, "duplicate_booking_handling")
        provider = SimpleNamespace(
            prefetched_uris=[],
            read_file_contents={},
            _transaction_handle=None,
        )

        class FakeVikingFS:
            async def ls(self, uri, output=None, ctx=None):
                assert uri == "viking://agent/a/memories/experiences"
                return [
                    {"uri": old_uri, "name": "booking_duplicate_handling.md"},
                    {
                        "uri": "viking://agent/a/memories/experiences/.overview.md",
                        "name": ".overview.md",
                    },
                ]

            async def read_file(self, uri, ctx=None):
                assert uri == old_uri
                return MemoryFileUtils.write(old_memory)

        decisions = await apply_admission_adapters(
            operations=ResolvedOperations(
                upsert_operations=[op],
                delete_file_contents=[],
                errors=[],
            ),
            adapters=[AgentExperienceAdmissionAdapter()],
            registry={},
            provider=provider,
            ctx=None,
            viking_fs=FakeVikingFS(),
            require_lock=False,
        )

        assert decisions[0].action == "redirect_update"
        assert decisions[0].reason == "near_experience_name"
        assert op.uris == [old_uri]
        assert op.old_memory_file_content.uri == old_uri
        assert op.memory_fields["experience_name"] == "booking_duplicate_handling"

    asyncio.run(run())


def test_agent_experience_admission_ignores_cross_parent_prefetched_candidates():
    async def run():
        other_uri = "viking://agent/other/memories/experiences/booking_duplicate_handling.md"
        new_uri = "viking://agent/a/memories/experiences/duplicate_booking_handling.md"
        other_memory = _experience(other_uri, "booking_duplicate_handling")
        op = _operation(new_uri, "duplicate_booking_handling")
        provider = SimpleNamespace(
            prefetched_uris=[other_uri],
            read_file_contents={other_uri: other_memory},
            _transaction_handle=None,
        )

        decisions = await apply_admission_adapters(
            operations=ResolvedOperations(
                upsert_operations=[op],
                delete_file_contents=[],
                errors=[],
            ),
            adapters=[AgentExperienceAdmissionAdapter()],
            registry={},
            provider=provider,
            ctx=None,
            viking_fs=None,
            require_lock=False,
        )

        assert decisions[0].action == "allow_create"
        assert decisions[0].candidate_uris == []
        assert decisions[0].trace["candidate_count"] == 0
        assert op.uris == [new_uri]
        assert op.old_memory_file_content is None

    asyncio.run(run())


def test_agent_experience_admission_skips_edits_and_supersedes():
    async def run():
        existing = _experience(
            "viking://agent/a/memories/experiences/cancel_order_flow.md",
            "cancel_order_flow",
        )
        edit_op = _operation(existing.uri, "cancel_order_flow")
        edit_op.old_memory_file_content = existing
        supersedes_op = _operation(
            "viking://agent/a/memories/experiences/broader_cancel_order_flow.md",
            "broader_cancel_order_flow",
            supersedes="cancel_order_flow",
        )
        provider = SimpleNamespace(
            prefetched_uris=[existing.uri],
            read_file_contents={existing.uri: existing},
            _transaction_handle=None,
        )

        decisions = await apply_admission_adapters(
            operations=ResolvedOperations(
                upsert_operations=[edit_op, supersedes_op],
                delete_file_contents=[],
                errors=[],
            ),
            adapters=[AgentExperienceAdmissionAdapter()],
            registry={},
            provider=provider,
            ctx=None,
            viking_fs=None,
            require_lock=False,
        )

        assert decisions == []
        assert edit_op.uris == [existing.uri]
        assert supersedes_op.old_memory_file_content is None

    asyncio.run(run())
