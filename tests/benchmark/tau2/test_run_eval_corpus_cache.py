from __future__ import annotations

import importlib.util
import json
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
