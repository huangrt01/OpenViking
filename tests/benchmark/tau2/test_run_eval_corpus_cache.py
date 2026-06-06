from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


def _load_run_eval():
    repo = Path(__file__).resolve().parents[3]
    scripts = repo / "benchmark" / "tau2" / "llm" / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "tau2_run_eval_under_test",
        scripts / "run_eval.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corpus_session_commit_concurrency_prefers_openviking_config():
    run_eval = _load_run_eval()

    assert (
        run_eval._corpus_session_commit_concurrency(
            {
                "benchmark": {"corpus_session_commit_concurrency": 2},
                "openviking": {"corpus_session_commit_concurrency": 4},
            }
        )
        == 4
    )
    assert (
        run_eval._corpus_session_commit_concurrency(
            {
                "benchmark": {"corpus_session_commit_concurrency": 3},
                "openviking": {},
            }
        )
        == 3
    )


@pytest.mark.parametrize("bad_value", [0, -1, "not-an-int"])
def test_corpus_session_commit_concurrency_rejects_invalid_values(bad_value):
    run_eval = _load_run_eval()

    with pytest.raises(ValueError, match="corpus_session_commit_concurrency"):
        run_eval._corpus_session_commit_concurrency(
            {
                "benchmark": {"corpus_session_commit_concurrency": bad_value},
                "openviking": {},
            }
        )


def test_cell_python_executable_prefers_python_bin(monkeypatch):
    run_eval = _load_run_eval()

    monkeypatch.setenv("PYTHON_BIN", "/tmp/custom-python")

    assert run_eval._cell_python_executable() == "/tmp/custom-python"


def test_tau2_subprocess_env_prioritizes_current_openviking_checkout(
    monkeypatch,
    tmp_path,
):
    run_eval = _load_run_eval()
    tau2_repo = tmp_path / "tau2"
    tau2_src = tau2_repo / "src"
    tau2_src.mkdir(parents=True)
    stale_openviking = "/tmp/stale-openviking"
    other_path = "/tmp/other"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([stale_openviking, other_path]))

    env = run_eval._tau2_subprocess_env(tau2_repo)

    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[:2] == [str(run_eval.REPO_ROOT), str(tau2_src)]
    assert entries[2:] == [stale_openviking, other_path]


def test_prepare_memory_corpus_reuses_cache_by_requested_commit_concurrency(tmp_path):
    run_eval = _load_run_eval()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    manifest_path = corpus_dir / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_transcript_format": "openviking_text",
                "train_include_system_prompt": False,
                "train_skip_failed_sessions": True,
                "train_tool_output_max_chars": 5000,
                "corpus_session_commit_concurrency": 8,
                "corpus_session_commit_worker_count": 2,
            }
        ),
        encoding="utf-8",
    )
    cell = {
        "domain": "retail",
        "strategy_id": "s1",
        "corpus_id": "c1",
        "corpus_key": "retail_c1",
        "corpus_dir": str(corpus_dir),
        "train_transcript_format": "openviking_text",
        "train_include_system_prompt": False,
        "train_skip_failed_sessions": True,
        "train_tool_output_max_chars": 5000,
        "corpus_session_commit_concurrency": 8,
    }

    row = run_eval._prepare_memory_corpus(cell, tmp_path, tmp_path / "out")

    assert row["reused"] is True
    assert row["corpus_session_commit_concurrency"] == 8
    assert row["corpus_session_commit_worker_count"] == 2

    mismatched = dict(cell, corpus_session_commit_concurrency=4)
    with pytest.raises(RuntimeError, match="corpus_session_commit_concurrency mismatch"):
        run_eval._prepare_memory_corpus(mismatched, tmp_path, tmp_path / "out2")


def test_prepare_memory_corpus_validates_outcome_mode_and_train_results_hash(tmp_path):
    run_eval = _load_run_eval()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    train_results_file = tmp_path / "train_results.json"
    train_results_file.write_text('{"simulations":[]}\n', encoding="utf-8")
    manifest_path = corpus_dir / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_transcript_format": "openviking_text",
                "train_outcome_mode": "label_only",
                "train_results_sha256": run_eval._file_sha256(train_results_file),
                "train_include_system_prompt": False,
                "train_skip_failed_sessions": False,
                "train_tool_output_max_chars": 5000,
                "corpus_session_commit_concurrency": 1,
                "openviking": {
                    "expected_agent_experience_failure_integration_mode": "prompt_guardrail"
                },
            }
        ),
        encoding="utf-8",
    )
    cell = {
        "domain": "airline",
        "strategy_id": "s1",
        "corpus_id": "c1",
        "corpus_key": "airline_c1",
        "corpus_dir": str(corpus_dir),
        "train_transcript_format": "openviking_text",
        "train_outcome_mode": "label_only",
        "train_results_file": str(train_results_file),
        "train_include_system_prompt": False,
        "train_skip_failed_sessions": False,
        "train_tool_output_max_chars": 5000,
        "corpus_session_commit_concurrency": 1,
        "agent_experience_failure_integration_mode": "prompt_guardrail",
    }

    row = run_eval._prepare_memory_corpus(cell, tmp_path, tmp_path / "out")

    assert row["reused"] is True

    mismatched_mode = dict(cell, train_outcome_mode="transcript_only")
    with pytest.raises(RuntimeError, match="train_outcome_mode mismatch"):
        run_eval._prepare_memory_corpus(mismatched_mode, tmp_path, tmp_path / "out2")

    mismatched_failure_mode = dict(
        cell,
        agent_experience_failure_integration_mode="metadata_only",
    )
    with pytest.raises(
        RuntimeError,
        match="agent_experience_failure_integration_mode mismatch",
    ):
        run_eval._prepare_memory_corpus(
            mismatched_failure_mode,
            tmp_path,
            tmp_path / "out_failure_mode",
        )

    changed_train_results = tmp_path / "changed_train_results.json"
    changed_train_results.write_text('{"simulations":[{"task_id":"x"}]}\n', encoding="utf-8")
    mismatched_train = dict(cell, train_results_file=str(changed_train_results))
    with pytest.raises(RuntimeError, match="train_results_sha256 mismatch"):
        run_eval._prepare_memory_corpus(mismatched_train, tmp_path, tmp_path / "out3")


def test_memory_constructor_mode_defaults_and_validates():
    run_eval = _load_run_eval()

    assert run_eval._memory_constructor_mode({"openviking": {}}, {}) == "full"
    assert (
        run_eval._memory_constructor_mode(
            {"openviking": {"memory_constructor_mode": "boundary_overlay"}},
            {},
        )
        == "boundary_overlay"
    )
    assert (
        run_eval._memory_constructor_mode(
            {"openviking": {"memory_constructor_mode": "full"}},
            {"memory_constructor_mode": "boundary_overlay"},
        )
        == "boundary_overlay"
    )
    with pytest.raises(ValueError, match="memory_constructor_mode"):
        run_eval._memory_constructor_mode(
            {"openviking": {}},
            {"memory_constructor_mode": "unknown"},
        )


def test_memory_applicability_gate_mode_defaults_and_validates():
    run_eval = _load_run_eval()

    assert run_eval._memory_applicability_gate_mode({"openviking": {}}, {}) == "none"
    assert (
        run_eval._memory_applicability_gate_mode(
            {"openviking": {"memory_applicability_gate_mode": "prewrite_action_overlap"}},
            {},
        )
        == "prewrite_action_overlap"
    )
    assert (
        run_eval._memory_applicability_gate_mode(
            {"openviking": {"memory_applicability_gate_mode": "none"}},
            {"memory_applicability_gate_mode": "prewrite_action_overlap"},
        )
        == "prewrite_action_overlap"
    )
    with pytest.raises(ValueError, match="memory_applicability_gate_mode"):
        run_eval._memory_applicability_gate_mode(
            {"openviking": {}},
            {"memory_applicability_gate_mode": "unknown"},
        )


def test_train_outcome_and_failed_retry_modes_validate():
    run_eval = _load_run_eval()

    assert run_eval._train_outcome_mode({}) == "transcript_only"
    assert run_eval._train_outcome_mode({"train_outcome_mode": "reward_info"}) == "reward_info"
    assert run_eval._failed_task_retry_count({"openviking": {}}, {}) == 0
    assert (
        run_eval._failed_task_retry_count(
            {"openviking": {"failed_task_retry_count": 2}},
            {},
        )
        == 2
    )
    assert (
        run_eval._failed_task_retry_count(
            {"openviking": {"failed_task_retry_count": 2}},
            {"failed_task_retry_count": 1},
        )
        == 1
    )
    assert run_eval._failed_task_retry_outcome_mode({}) == "reward_info"
    assert (
        run_eval._failed_task_retry_outcome_mode(
            {"failed_task_retry_outcome_mode": "label_only"}
        )
        == "label_only"
    )
    with pytest.raises(ValueError, match="train_outcome_mode"):
        run_eval._train_outcome_mode({"train_outcome_mode": "bad"})
    with pytest.raises(ValueError, match="failed_task_retry_count"):
        run_eval._failed_task_retry_count({"openviking": {}}, {"failed_task_retry_count": -1})
    with pytest.raises(ValueError, match="failed_task_retry_outcome_mode"):
        run_eval._failed_task_retry_outcome_mode({"failed_task_retry_outcome_mode": "bad"})


def test_memory_corpus_key_isolates_eval_memory_writes_by_repeat():
    run_eval = _load_run_eval()
    strategy = {"id": "s1", "corpus_id": "c1"}

    assert (
        run_eval._memory_corpus_key_for(
            domain="airline",
            strategy=strategy,
            train_num_tasks=None,
        )
        == "airline_c1"
    )
    assert (
        run_eval._memory_corpus_key_for(
            domain="airline",
            strategy=strategy,
            train_num_tasks=3,
            repeat_index=2,
            repeat_isolated=True,
        )
        == "airline_c1_train3_r2"
    )
    with pytest.raises(ValueError, match="repeat_index"):
        run_eval._memory_corpus_key_for(
            domain="airline",
            strategy=strategy,
            train_num_tasks=None,
            repeat_isolated=True,
        )


def test_run_plan_isolates_failed_retry_corpora_by_repeat(tmp_path):
    run_eval = _load_run_eval()
    config = {
        "benchmark": {
            "domains": ["airline"],
            "train_split_name": "train",
            "eval_split_name": "test",
            "repeat_count": 2,
            "seed": 300,
            "max_steps": 200,
            "task_max_concurrency": 1,
        },
        "eval": {
            "require_fixed_first_user": False,
            "user_simulator_policy": "official",
        },
        "model": {
            "agent_llm": "agent-model",
            "user_llm": "user-model",
        },
        "openviking": {
            "url": "http://127.0.0.1:9999",
            "account": "acct",
            "timeout_seconds": 600,
            "wait_timeout_seconds": 600,
            "reuse_corpus_across_runs": True,
        },
        "paths": {
            "tau2_repo": str(tmp_path / "tau2"),
            "output_dir": str(tmp_path / "result"),
            "corpus_cache_dir": str(tmp_path / "corpora"),
        },
        "strategies": [
            {
                "id": "memory_read_only",
                "memory_backend": "openviking",
                "train_memory_mode": "experience_only",
                "corpus_id": "shared",
            },
            {
                "id": "memory_retry",
                "memory_backend": "openviking",
                "train_memory_mode": "experience_only",
                "corpus_id": "retry",
                "failed_task_retry_count": 2,
            },
        ],
    }

    plan = run_eval._build_plan(
        config,
        "run1",
        selected_domains=None,
        selected_strategy_ids=None,
        task_ids=None,
        num_tasks=1,
        train_num_tasks=None,
        repeat_count_override=None,
        cell_concurrency_override=None,
        strategy_concurrency_override=None,
    )

    cells = {(cell["strategy_id"], cell["repeat_index"]): cell for cell in plan["cells"]}
    read_r1 = cells[("memory_read_only", 1)]
    read_r2 = cells[("memory_read_only", 2)]
    retry_r1 = cells[("memory_retry", 1)]
    retry_r2 = cells[("memory_retry", 2)]

    assert read_r1["corpus_key"] == "airline_shared"
    assert read_r2["corpus_key"] == "airline_shared"
    assert read_r1["repeat_isolated_corpus"] is False
    assert read_r2["repeat_isolated_corpus"] is False
    assert retry_r1["corpus_key"] == "airline_retry_r1"
    assert retry_r2["corpus_key"] == "airline_retry_r2"
    assert retry_r1["eval_memory_writes"] is True
    assert retry_r2["repeat_isolated_corpus"] is True

    for cell in [read_r1, read_r2, retry_r1, retry_r2]:
        command = cell["command"]
        corpus_dir_index = command.index("--corpus-dir")
        account_index = command.index("--openviking-account")
        assert command[corpus_dir_index + 1] == cell["corpus_dir"]
        assert command[account_index + 1] == f"acct-{cell['corpus_key']}"


def test_tau2_command_passes_memory_constructor_mode(tmp_path):
    run_eval = _load_run_eval()
    config = {
        "benchmark": {
            "train_split_name": "train",
            "eval_split_name": "test",
            "max_steps": 200,
            "task_max_concurrency": 1,
        },
        "model": {
            "agent_llm": "agent-model",
            "user_llm": "user-model",
        },
        "openviking": {
            "url": "http://127.0.0.1:9999",
            "account": "acct",
            "timeout_seconds": 600,
            "wait_timeout_seconds": 600,
            "reuse_corpus_across_runs": True,
        },
        "paths": {
            "tau2_repo": str(tmp_path / "tau2"),
            "output_dir": str(tmp_path / "result"),
            "corpus_cache_dir": str(tmp_path / "corpora"),
        },
    }
    strategy = {
        "id": "s1",
        "memory_backend": "openviking",
        "train_memory_mode": "experience_only",
        "corpus_id": "c1",
        "memory_constructor_mode": "boundary_overlay",
        "memory_applicability_gate_mode": "prewrite_action_overlap",
        "train_outcome_mode": "reward_info",
        "failed_task_retry_count": 2,
        "failed_task_retry_outcome_mode": "reward_info",
        "agent_experience_failure_integration_mode": "comparative_insight",
    }

    command = run_eval._tau2_command(
        config,
        domain="airline",
        strategy=strategy,
        configured_run_id="run1",
        run_label="cell1",
        repeat_index=1,
        task_ids=None,
        num_tasks=1,
        train_num_tasks=None,
        seed=300,
    )

    assert command is not None
    index = command.index("--memory-constructor-mode")
    assert command[index + 1] == "boundary_overlay"
    index = command.index("--memory-applicability-gate-mode")
    assert command[index + 1] == "prewrite_action_overlap"
    index = command.index("--train-outcome-mode")
    assert command[index + 1] == "reward_info"
    index = command.index("--failed-task-retry-count")
    assert command[index + 1] == "2"
    index = command.index("--failed-task-retry-outcome-mode")
    assert command[index + 1] == "reward_info"
    index = command.index("--expected-agent-experience-failure-integration-mode")
    assert command[index + 1] == "comparative_insight"
