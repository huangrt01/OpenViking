from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_runner_module():
    repo = Path(__file__).resolve().parents[3]
    script = repo / "benchmark" / "tau2" / "llm" / "scripts" / "run_memory_v2_eval.py"
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("tau2_memory_v2_eval_under_test", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Match:
    def __init__(self, uri: str, *, abstract: str = "", overview: str = ""):
        self.uri = uri
        self.abstract = abstract
        self.overview = overview


class _TaskClient:
    def __init__(self, task):
        self.task = task

    def get_task(self, task_id: str):
        return self.task


def test_wait_task_fails_fast_on_openviking_error_status():
    module = _load_runner_module()

    with pytest.raises(RuntimeError, match="Task not found"):
        module._wait_task(
            _TaskClient(
                {
                    "status": "error",
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "Task not found or expired",
                    },
                }
            ),
            "task-1",
            timeout=3600,
        )


def test_memory_extract_skipped_from_task_reads_telemetry_summary():
    module = _load_runner_module()

    task = {
        "telemetry": {
            "summary": {
                "memory": {
                    "extract": {
                        "actions": {
                            "created": 1,
                            "merged": 2,
                            "skipped": 3,
                        }
                    }
                }
            }
        }
    }

    assert module._memory_extract_skipped_from_task(task) == 3
    assert module._memory_extract_skipped_from_task({"telemetry": {}}) == 0


def test_corpus_provenance_records_train_config_and_git_identity(tmp_path, monkeypatch):
    module = _load_runner_module()
    train_results = tmp_path / "train_results.json"
    train_results.write_text('{"simulations":[]}', encoding="utf-8")
    config_file = tmp_path / "ov.conf"
    config_file.write_text('{"memory":{"agent_memory_enabled":true}}', encoding="utf-8")

    monkeypatch.setenv("OPENVIKING_CONFIG_FILE", str(config_file))
    monkeypatch.setattr(
        module,
        "_git_head_commit",
        lambda repo: f"commit:{Path(repo).name}",
    )

    provenance = module._corpus_provenance(
        SimpleNamespace(tau2_repo=tmp_path / "tau2"),
        train_results,
    )

    assert provenance["train_results_sha256"] == module._file_sha256(train_results)
    assert provenance["tau2"] == {
        "repo": str(tmp_path / "tau2"),
        "commit": "commit:tau2",
    }
    assert provenance["openviking"]["repo"] == str(module.REPO_ROOT)
    assert provenance["openviking"]["commit"] == f"commit:{module.REPO_ROOT.name}"
    assert provenance["openviking"]["config_file"] == str(config_file)
    assert provenance["openviking"]["config_file_sha256"] == module._file_sha256(config_file)


def test_server_memory_config_report_validates_failure_integration_mode(tmp_path, monkeypatch):
    module = _load_runner_module()
    config_file = tmp_path / "ov.conf"
    config_file.write_text(
        '{"memory":{"agent_experience_failure_integration_mode":"prompt_guardrail"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENVIKING_CONFIG_FILE", str(config_file))

    report = module._server_memory_config_report(
        SimpleNamespace(expected_agent_experience_failure_integration_mode="prompt_guardrail")
    )

    assert report["checked"] is True
    assert report["agent_experience_failure_integration_mode"] == "prompt_guardrail"
    with pytest.raises(RuntimeError, match="failure_integration_mode mismatch"):
        module._server_memory_config_report(
            SimpleNamespace(expected_agent_experience_failure_integration_mode="metadata_only")
        )


def test_cached_corpus_rejects_train_results_sha_mismatch(tmp_path):
    module = _load_runner_module()
    train_results = tmp_path / "train_results.json"
    train_results.write_text('{"simulations":[]}', encoding="utf-8")
    corpus_manifest = tmp_path / "corpus_manifest.json"
    corpus_manifest.write_text(
        json.dumps(
            {
                "domain": "retail",
                "train_results_sha256": "not-the-current-sha",
                "train_transcript_format": module.TRAIN_TRANSCRIPT_OPENVIKING_TEXT,
                "train_include_system_prompt": False,
                "train_tool_output_max_chars": module.DEFAULT_TRAIN_TOOL_OUTPUT_MAX_CHARS,
                "train_skip_failed_sessions": False,
                "corpus_session_commit_concurrency": 1,
                "committed_session_count": 1,
                "memories_extracted_total": 1,
                "memory_extract_skipped_total": 0,
                "corpus_probe": {"match_count": 1, "read_non_empty_count": 1},
                "memory_graph_health": {
                    "healthy": True,
                    "memory_type_counts": {"experiences": 1, "trajectories": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        corpus_session_commit_concurrency=1,
        force_train=False,
        train_transcript_format=module.TRAIN_TRANSCRIPT_OPENVIKING_TEXT,
        train_include_system_prompt=False,
        train_tool_output_max_chars=module.DEFAULT_TRAIN_TOOL_OUTPUT_MAX_CHARS,
        train_skip_failed_sessions=False,
        domain="retail",
    )

    with pytest.raises(ValueError, match="train_results_sha256 mismatch"):
        module._train(args, train_results, corpus_manifest)


def test_memory_graph_root_uri_uses_memories_path_segment():
    module = _load_runner_module()

    assert (
        module._memory_graph_root_uri("viking://agent/memories-agent/memories/experiences")
        == "viking://agent/memories-agent/memories"
    )


def test_concrete_memory_matches_skip_overviews_and_directories():
    module = _load_runner_module()

    matches = module._concrete_memory_matches(
        [
            _Match("viking://agent/a/memories/experiences/foo.md"),
            _Match("viking://agent/a/memories/experiences/foo.overview.md"),
            _Match("viking://agent/a/memories/experiences/foo.abstract.md"),
            _Match("viking://agent/a/memories/experiences"),
        ]
    )

    assert [match.uri for match in matches] == ["viking://agent/a/memories/experiences/foo.md"]


def test_corpus_probe_read_non_empty_requires_successful_read():
    module = _load_runner_module()

    class FakeSearchResult:
        memories = [
            _Match(
                "viking://agent/a/memories/experiences/foo.md",
                abstract="fallback abstract",
            )
        ]

    class FakeClient:
        def search(self, **kwargs):
            return FakeSearchResult()

        def read(self, uri: str):
            raise RuntimeError("read failed")

    probe = module._probe_corpus(
        SimpleNamespace(domain="retail", search_uri="viking://agent/a/memories", retrieval_top_k=2),
        FakeClient(),
    )

    assert probe["match_count"] == 1
    assert probe["read_non_empty_count"] == 0
    assert probe["matches"][0]["readable"] is False
    assert probe["matches"][0]["text_chars"] > 0
    assert "RuntimeError: read failed" in probe["matches"][0]["read_error"]


def test_corpus_probe_gate_requires_readable_concrete_memory():
    module = _load_runner_module()

    with pytest.raises(RuntimeError, match="no concrete memory files"):
        module._raise_if_unhealthy_corpus_probe({"match_count": 0, "read_non_empty_count": 0})

    with pytest.raises(RuntimeError, match="non-empty concrete memory files"):
        module._raise_if_unhealthy_corpus_probe({"match_count": 2, "read_non_empty_count": 0})

    module._raise_if_unhealthy_corpus_probe({"match_count": 2, "read_non_empty_count": 1})


def test_memory_graph_gate_requires_concrete_linked_memory():
    module = _load_runner_module()

    module._raise_if_unhealthy_memory_graph(
        {
            "healthy": True,
            "memory_type_counts": {"experiences": 2, "trajectories": 1},
        }
    )

    with pytest.raises(RuntimeError, match="trajectories=0"):
        module._raise_if_unhealthy_memory_graph(
            {
                "healthy": True,
                "memory_type_counts": {"experiences": 2},
            }
        )

    with pytest.raises(RuntimeError, match="source_linkless_experience_count=1"):
        module._raise_if_unhealthy_memory_graph(
            {
                "healthy": False,
                "memory_type_counts": {"experiences": 2, "trajectories": 1},
                "source_linkless_experience_count": 1,
            }
        )


def test_corpus_evidence_summary_rejects_legacy_or_empty_manifest():
    module = _load_runner_module()

    legacy = {
        "committed_session_count": 3,
        "memories_extracted_total": 4,
    }
    evidence = module._corpus_evidence_summary(legacy)
    assert evidence["claim_valid"] is False
    assert "missing_corpus_probe" in evidence["invalid_reasons"]
    assert "missing_memory_graph_health" in evidence["invalid_reasons"]

    valid = {
        "committed_session_count": 3,
        "memories_extracted_total": 4,
        "memory_extract_skipped_total": 0,
        "corpus_probe": {"match_count": 2, "read_non_empty_count": 1},
        "memory_graph_health": {
            "healthy": True,
            "memory_type_counts": {"experiences": 2, "trajectories": 1},
        },
    }
    assert module._corpus_evidence_summary(valid)["claim_valid"] is True

    valid["memory_extract_skipped_total"] = 1
    skipped = module._corpus_evidence_summary(valid)
    assert skipped["claim_valid"] is False
    assert "memory_extract_skipped_total=1" in skipped["invalid_reasons"]


def test_retrieval_trace_and_effect_evidence_mark_zero_retrieval_invalid(tmp_path):
    module = _load_runner_module()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "decision_node": "first_user",
                "match_count": 0,
                "injected_count": 0,
                "matches": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    trace_summary = module._retrieval_trace_summary(trace_path)
    assert trace_summary["event_count"] == 1
    assert trace_summary["match_count_sum"] == 0

    evidence = module._effect_evidence_summary({"simulation_count": 2}, trace_summary)
    assert evidence["claim_valid"] is False
    assert "retrieval_trace.match_count_sum=0" in evidence["invalid_reasons"]
    assert "retrieval_trace.injected_count_sum=0" in evidence["invalid_reasons"]

    with pytest.raises(RuntimeError, match="do not include this run in effect tables"):
        module._raise_if_invalid_effect_evidence(evidence)


def test_effect_evidence_gate_accepts_matched_injected_retrieval(tmp_path):
    module = _load_runner_module()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "match_count": 2,
                "injected_count": 1,
                "matches": [
                    {"uri": "viking://user/example/memories/events/1.md", "text_chars": 25}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    trace_summary = module._retrieval_trace_summary(trace_path)
    evidence = module._effect_evidence_summary({"simulation_count": 1}, trace_summary)

    assert evidence["claim_valid"] is True
    module._raise_if_invalid_effect_evidence(evidence)


def test_reward_info_outcome_message_preserves_tau2_reward_details():
    module = _load_runner_module()

    text = module._outcome_message(
        {
            "task_id": "32",
            "trial": 0,
            "reward_info": {
                "reward": 0.0,
                "db_check": {"score": False, "diff": {"reservation": "changed"}},
                "nl_assertions": [{"score": False, "message": "Wrong cabin"}],
            },
        },
        module.TRAIN_OUTCOME_REWARD_INFO,
    )

    assert "tau2_reward_info_json:" in text
    payload = json.loads(text.rsplit("tau2_reward_info_json:", 1)[1])
    assert payload["task_id"] == "32"
    assert payload["outcome_label"] == "failure"
    assert payload["reward"] == 0.0
    assert payload["db_match"] is False
    assert payload["reward_info"]["nl_assertions"][0]["message"] == "Wrong cabin"


def test_replace_simulations_with_retry_keeps_non_retried_tasks():
    module = _load_runner_module()
    current = {
        "simulations": [
            {"task_id": "18", "trial": 0, "reward_info": {"reward": 1.0}},
            {"task_id": "32", "trial": 0, "reward_info": {"reward": 0.0}},
        ]
    }
    retry = {
        "simulations": [
            {
                "task_id": "32",
                "trial": 0,
                "reward_info": {"reward": 1.0, "db_check": {"score": True}},
            }
        ]
    }

    merged, replaced = module._replace_simulations_with_retry(current, retry)

    assert [sim["task_id"] for sim in merged["simulations"]] == ["18", "32"]
    assert module._reward(merged["simulations"][0]) == 1.0
    assert module._reward(merged["simulations"][1]) == 1.0
    assert replaced == [
        {
            "task_id": "32",
            "trial": 0,
            "before_reward": 0.0,
            "before_db_match": None,
            "after_reward": 1.0,
            "after_db_match": True,
        }
    ]


def test_retrieval_budget_defaults_preserve_explicit_zero_inject_limit():
    module = _load_runner_module()
    args = SimpleNamespace(
        retrieval_top_k=4,
        first_user_retrieval_top_k=None,
        first_user_inject_top_k=0,
        prewrite_retrieval_top_k=None,
        prewrite_inject_top_k=None,
    )

    module._fill_retrieval_budget_defaults(args)

    assert args.first_user_retrieval_top_k == 4
    assert args.first_user_inject_top_k == 0
    assert args.prewrite_retrieval_top_k == 4
    assert args.prewrite_inject_top_k == 4


def test_boundary_overlay_constructor_keeps_situation_and_reflect_only():
    module = _load_runner_module()

    constructed, trace = module._construct_memory_text(
        """
## Situation
The user asks for a staged reservation update.

## Approach
Call the full modification procedure.

## Reflect
Do not collapse separately confirmed mutations into one tool call.
""",
        module.MEMORY_CONSTRUCTOR_BOUNDARY_OVERLAY,
    )

    assert "## Situation" in constructed
    assert "staged reservation update" in constructed
    assert "## Reflect" in constructed
    assert "Do not collapse" in constructed
    assert "## Approach" not in constructed
    assert trace["constructor_applied"] is True
    assert trace["constructor_sections"] == ["Situation", "Reflect"]


def test_boundary_overlay_constructor_falls_back_when_sections_missing():
    module = _load_runner_module()

    constructed, trace = module._construct_memory_text(
        "Plain memory without expected headings.",
        module.MEMORY_CONSTRUCTOR_BOUNDARY_OVERLAY,
    )

    assert constructed == "Plain memory without expected headings."
    assert trace["constructor_applied"] is False
    assert trace["constructor_sections"] == []


def test_prewrite_action_overlap_gate_passes_matching_memory():
    module = _load_runner_module()

    trace = module._memory_applicability_gate(
        "## Reflect\nValidate the requested direct flight before changing the reservation.",
        uri="viking://user/example/memories/experiences/flight_nonstop_modification_check.md",
        mode=module.MEMORY_APPLICABILITY_GATE_PREWRITE_ACTION_OVERLAP,
        decision_node="before_write_tool_call",
        tool_calls=[
            {
                "name": "update_reservation_flights",
                "arguments": {"flights": [{"flight_number": "HAT041"}]},
            }
        ],
    )

    assert trace["applicability_gate_applied"] is True
    assert trace["applicability_gate_passed"] is True
    assert trace["applicability_gate_reason"] == "action_token_overlap"
    assert "flight_update" in trace["applicability_gate_overlap_tokens"]


def test_prewrite_action_overlap_gate_filters_non_matching_memory():
    module = _load_runner_module()

    trace = module._memory_applicability_gate(
        "## Reflect\nDo not offer compensation before the user explicitly asks for it.",
        uri="viking://user/example/memories/experiences/compensation_request_handoff.md",
        mode=module.MEMORY_APPLICABILITY_GATE_PREWRITE_ACTION_OVERLAP,
        decision_node="before_write_tool_call",
        tool_calls=[
            {
                "name": "update_reservation_baggages",
                "arguments": {"total_baggages": 2},
            }
        ],
    )

    assert trace["applicability_gate_applied"] is True
    assert trace["applicability_gate_passed"] is False
    assert trace["applicability_gate_reason"] == "no_action_token_overlap"
    assert "baggage_update" in trace["applicability_gate_action_tokens"]


def test_prewrite_action_overlap_gate_ignores_generic_flight_in_cancellation_memory():
    module = _load_runner_module()

    trace = module._memory_applicability_gate(
        "## Situation\nThe user asks whether a flight cancellation is refundable.",
        uri="viking://user/example/memories/experiences/flight_cancellation_eligibility_check.md",
        mode=module.MEMORY_APPLICABILITY_GATE_PREWRITE_ACTION_OVERLAP,
        decision_node="before_write_tool_call",
        tool_calls=[
            {
                "name": "update_reservation_flights",
                "arguments": {"flights": [{"flight_number": "HAT041"}]},
            }
        ],
    )

    assert trace["applicability_gate_passed"] is False
    assert trace["applicability_gate_action_tokens"] == ["flight_update"]
    assert "cancel" in trace["applicability_gate_memory_cues"]
    assert "flight_update" not in trace["applicability_gate_memory_cues"]


def test_prewrite_action_overlap_gate_does_not_filter_first_user():
    module = _load_runner_module()

    trace = module._memory_applicability_gate(
        "Anything",
        mode=module.MEMORY_APPLICABILITY_GATE_PREWRITE_ACTION_OVERLAP,
        decision_node="first_user",
        tool_calls=[],
    )

    assert trace["applicability_gate_applied"] is False
    assert trace["applicability_gate_passed"] is True
    assert trace["applicability_gate_reason"] == "non_prewrite_node"
