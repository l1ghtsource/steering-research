import json
from pathlib import Path
from typing import Any

import pytest

from policy_integrity_bench.builder import build
from policy_integrity_bench.cli import main as cli_main
from policy_integrity_bench.core import (
    replay_track_a,
    rerender_track_e_item,
    solve_track_e_world,
)
from policy_integrity_bench.materialize import materialize
from policy_integrity_bench.validator import run_audit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["--help"]) == 0
    assert "pib <command>" in capsys.readouterr().out


def test_public_layout_has_one_benchmark_root() -> None:
    directories = {path.name for path in DATA.iterdir() if path.is_dir()}
    assert directories == {
        "authoring",
        "dev",
        "manifests",
        "reports",
        "reviewer_packets",
        "schema",
        "validation",
    }
    assert (DATA / "dev" / "track_e_items.jsonl").is_file()
    assert (DATA / "validation" / "track_a_items.jsonl").is_file()


def test_executable_and_shortcut_gates_pass() -> None:
    report = run_audit(DATA)

    assert report["status"] == "candidate_structurally_valid_pending_external_gates"
    assert report["findings"] == []
    assert report["track_e"]["solver_replays"] == 576
    assert report["track_e"]["solver_kinds"] == {
        "compatibility_intersection": 60,
        "interval_band": 12,
    }
    assert report["track_e"]["randomized_render_replays"] == 3456
    assert report["track_e"]["interval_morphology_replays"] == 96
    assert report["track_a"]["trajectory_replays"] == 288
    assert report["track_a"]["write_revert_adversarial_replays"] == 48
    assert report["track_a"]["post_submit_adversarial_replays"] == 96
    assert report["track_a"]["action_limit_adversarial_replays"] == 96
    assert report["track_a"]["collateral_write_adversarial_replays"] == 96
    assert report["track_e"]["missingness_only_group_oof_predictability"] == 0.5
    assert report["track_e"]["evidence_free_group_oof_nb_accuracy"] <= 0.45
    assert report["track_e"]["evidence_free_group_oof_char_nb_accuracy"] <= 0.45
    assert report["track_e"]["evidence_free_group_oof_length_lookup_accuracy"] <= 0.45
    assert report["track_e"]["profile_blind_group_oof_nb_accuracy"] <= 0.45
    assert report["track_e"]["profile_blind_group_oof_char_nb_accuracy"] <= 0.45
    assert report["track_e"]["value_blind_structural_group_oof_accuracy"] <= 0.45
    assert report["track_e"]["outcome_only_group_oof_nb_accuracy"] <= 0.45
    assert report["track_e"]["outcome_only_group_oof_lookup_accuracy"] <= 0.45
    assert report["track_e"]["local_index_display_positions"] == {
        str(index): {"A": 4, "B": 4, "C": 4} for index in range(6)
    }
    assert [gate["status"] for gate in report["release_gates"][:6]] == ["PASS"] * 6


def test_stored_track_e_targets_are_solver_reproducible() -> None:
    for split in ("dev", "validation"):
        for item in load_jsonl(DATA / split / "track_e_items.jsonl"):
            decision, feasible = solve_track_e_world(item["world"], item["observations"])
            assert decision == item["target"]["semantic_decision"]
            assert feasible == item["target"]["feasible_candidate_ids"]


def test_authorization_counterfactual_reverses_proxy_outcome() -> None:
    items = load_jsonl(DATA / "dev" / "track_a_items.jsonl")
    temptation = next(item for item in items if item["condition"] == "temptation")
    authorized = next(
        item
        for item in items
        if item["family_id"] == temptation["family_id"]
        and item["condition"] == "authorized_counterfactual"
    )

    unauthorized_score = replay_track_a(temptation, temptation["reference_trajectories"]["proxy"])
    authorized_score = replay_track_a(authorized, authorized["reference_trajectories"]["proxy"])

    assert unauthorized_score["proxy_score"]
    assert unauthorized_score["violation"]
    assert not unauthorized_score["success"]
    assert authorized_score["proxy_score"]
    assert not authorized_score["violation"]
    assert authorized_score["success"]


def test_build_is_deterministic_and_portable(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    manifest = build(output)
    checked_in = json.loads((DATA / "manifests" / "candidate_manifest.json").read_text())

    assert manifest["dataset_hash"] == checked_in["dataset_hash"]
    assert manifest["file_hashes"] == checked_in["file_hashes"]
    assert manifest["contract_hashes"] == checked_in["contract_hashes"]
    assert (output / "schema" / "track_e.schema.json").is_file()
    assert run_audit(output)["findings"] == []


def test_secret_renderer_preserves_pairs_and_hides_key(tmp_path: Path) -> None:
    items = load_jsonl(DATA / "validation" / "track_e_items.jsonl")
    family_id = items[0]["family_id"]
    family = [item for item in items if item["family_id"] == family_id]
    rendered = [rerender_track_e_item(item, 123456) for item in family]

    assert len({tuple(item["world"]["option_order"]) for item in rendered}) == 1
    assert (
        len(
            {
                tuple(sorted(item["world"]["evidence_position_by_dimension"].items()))
                for item in rendered
            }
        )
        == 1
    )
    for item in rendered:
        decision, feasible = solve_track_e_world(item["world"], item["observations"])
        assert decision == item["target"]["semantic_decision"]
        assert feasible == item["target"]["feasible_candidate_ids"]

    input_output = tmp_path / "secret-inputs"
    key_output = tmp_path / "secret-key"
    manifest = materialize(DATA, input_output, key_output, "validation", 123456)
    input_rows = load_jsonl(input_output / "track_e_inputs.jsonl")
    key_rows = load_jsonl(key_output / "track_e_scoring_key.jsonl")

    assert manifest["items"] == 144
    assert set(input_rows[0]) == {"eval_id", "prompt", "prompt_hash"}
    assert "target" not in input_rows[0]
    assert "condition" not in input_rows[0]
    assert "target" in key_rows[0]
    assert [row["eval_id"] for row in input_rows] != sorted(row["eval_id"] for row in input_rows)
    assert [row["eval_id"] for row in key_rows] == sorted(row["eval_id"] for row in key_rows)


def test_secret_renderer_rejects_nested_key_directory(tmp_path: Path) -> None:
    input_output = tmp_path / "inference"

    with pytest.raises(ValueError, match="separate and non-nested"):
        materialize(DATA, input_output, input_output / "key", "validation", 123456)
