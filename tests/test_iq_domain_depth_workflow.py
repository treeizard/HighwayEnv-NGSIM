from __future__ import annotations

from pathlib import Path

from scripts_gail.iq_study import CONFIRMATIONS, matrix_cells, read_locked_recipe, trainer_command
from scripts_gail.run_iq_convergence_pilot import candidates, pilot_recipe


def test_iq_matrix_is_confirmation_first_and_has_12_unique_cells():
    cells = matrix_cells(confirmation_first=True)
    assert tuple(cells[:2]) == CONFIRMATIONS
    assert len(cells) == 12
    assert len(set(cells)) == 12
    assert {domain for domain, _depth, _seed in cells} == {"us", "japanese"}
    assert {depth for _domain, depth, _seed in cells} == {2, 3}
    assert {seed for _domain, _depth, seed in cells} == {0, 1, 2}


def test_pilot_grid_has_eight_deterministic_reference_candidates():
    grid = candidates()
    assert len(grid) == 8
    assert len({tuple(sorted(row.items())) for row in grid}) == 8
    assert {row["entropy_temperature"] for row in grid} == {0.001, 0.01}
    assert {row["q_learning_rate"] for row in grid} == {3.0e-5, 1.0e-4}


def test_trainer_command_locks_online_recurrent_architecture(tmp_path):
    args = type("Args", (), {
        "pilot_updates": 120, "pilot_joint_updates": 100, "pilot_eval_every": 40,
        "pilot_q_only_updates": 20, "pilot_initial_replay": 512, "pilot_collect_steps": 256,
    })()
    recipe = pilot_recipe(candidates()[0], args)
    command = trainer_command(
        recipe, domain="japanese", depth=3, seed=1,
        expert_data=tmp_path / "expert", episode_root=tmp_path / "episodes",
        initial_checkpoint=tmp_path / "bc.pt", out_dir=tmp_path / "out", device="cuda",
    )
    joined = " ".join(command)
    assert "--domain japanese --scene japanese" in joined
    assert "--transformer-layers 3" in joined
    assert "--transformer-dropout 0.0" in joined
    assert "--memory-context-length 32" in joined
    assert "--training-context-length 8" in joined
    assert "--sequence-length 8" in joined
    assert "--validation-sequence-length 32" in joined
    assert "--sequences-per-update 4" in joined
    assert "--chi2-regularization expert" in joined
    assert "--no-training-enable-collision" in joined


def test_locked_recipe_validation(tmp_path):
    args = type("Args", (), {
        "pilot_updates": 120, "pilot_joint_updates": 100, "pilot_eval_every": 40,
        "pilot_q_only_updates": 20, "pilot_initial_replay": 512, "pilot_collect_steps": 256,
    })()
    recipe = pilot_recipe(candidates()[0], args)
    recipe["status"] = "locked"
    path = tmp_path / "recipe.json"
    import json
    path.write_text(json.dumps(recipe), encoding="utf-8")
    assert read_locked_recipe(path)["optimization"]["chi2_regularization"] == "expert"
