import json
import subprocess
import sys
import unittest
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from simulator import run_simulation  # noqa: E402
from utils.schema import SCHEMA_VERSION, validate_result_schema  # noqa: E402
from validate_results import validate_file  # noqa: E402


class SchemaValidationTest(unittest.TestCase):
    def test_single_run_schema_valid(self):
        results = run_simulation(num_agents=5, runs=1, seed=123, conformity_threshold=0.6, whale_ratio=0.2, output=None)
        self.assertEqual(len(results), 1)
        result = results[0]
        # Ensures schema discipline and presence.
        validate_result_schema(result)
        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertIn("run_id", result)
        self.assertIn("timestamp", result)
        self.assertEqual(result["num_agents"], 5)

    def test_validate_file_passes_for_generated_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "runs.jsonl"
            run_simulation(num_agents=5, runs=2, seed=99, conformity_threshold=0.5, whale_ratio=0.3, output=str(out_path))
            checked, errors = validate_file(out_path)
            self.assertGreaterEqual(checked, 2)
            self.assertEqual(errors, [])

    def test_validate_file_catches_invalid_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "bad.jsonl"
            bad_path.write_text('{"foo": "bar"}\nnot json\n', encoding="utf-8")
            checked, errors = validate_file(bad_path)
            self.assertEqual(checked, 0)
            self.assertGreaterEqual(len(errors), 2)

    def test_validate_cli_failure_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "bad.jsonl"
            bad_path.write_text('not json\n{"schema_version": "0.2"}\n', encoding="utf-8")
            cmd = [sys.executable, str(PROJECT_ROOT / "src" / "validate_results.py"), "--file", str(bad_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("Line 1: invalid JSON", proc.stdout)
            self.assertIn("Line 2: Missing required fields", proc.stdout)

    def test_determinism_same_seed_and_drift_detection(self):
        baseline = run_simulation(num_agents=5, runs=1, seed=21, conformity_threshold=0.6, whale_ratio=0.2, output=None)[0]
        repeat = run_simulation(num_agents=5, runs=1, seed=21, conformity_threshold=0.6, whale_ratio=0.2, output=None)[0]
        different_seed = run_simulation(num_agents=5, runs=1, seed=22, conformity_threshold=0.6, whale_ratio=0.2, output=None)[0]

        baseline_clean = self._canonicalize_result(baseline)
        repeat_clean = self._canonicalize_result(repeat)
        different_clean = self._canonicalize_result(different_seed)

        self.assertEqual(baseline_clean, repeat_clean, "Same seed should produce identical canonicalized output")
        self.assertNotEqual(baseline_clean, different_clean, "Different seed should produce drift-detectable output")
        self.assertEqual(baseline["cascade_rounds"], repeat["cascade_rounds"])
        self.assertEqual(baseline["agents_flipped_total"], repeat["agents_flipped_total"])

    def test_schema_rejects_unexpected_fields(self):
        result = run_simulation(num_agents=3, runs=1, seed=7, conformity_threshold=0.6, whale_ratio=0.2, output=None)[0]
        result_with_extra = dict(result)
        result_with_extra["extra_field"] = "oops"
        with self.assertRaises(ValueError):
            validate_result_schema(result_with_extra)
        agent_extra = dict(result["agents"][0])
        agent_extra["unexpected"] = True
        bad_agent_result = dict(result)
        bad_agent_result["agents"] = [agent_extra] + result["agents"][1:]
        with self.assertRaises(ValueError):
            validate_result_schema(bad_agent_result)

    def test_schema_snapshot_matches_fixture(self):
        fixture_path = Path(__file__).parent / "fixtures" / "snapshot_v0_4.json"
        result = run_simulation(num_agents=5, runs=1, seed=11, conformity_threshold=0.6, whale_ratio=0.2, output=None)[0]
        canonical = self._canonicalize_result(result)
        fixture = self._load_fixture(fixture_path)
        self.assertEqual(canonical, fixture)

    def test_cascade_rounds_and_flips_bounds(self):
        result = run_simulation(num_agents=6, runs=1, seed=33, conformity_threshold=0.6, whale_ratio=0.2, output=None)[0]
        self.assertGreaterEqual(result["cascade_rounds"], 1)
        self.assertLessEqual(result["cascade_rounds"], 5)
        self.assertIsInstance(result["agents_flipped_total"], int)
        self.assertGreaterEqual(result["agents_flipped_total"], 0)

    def test_migration_to_v0_4_defaults(self):
        legacy = {
            "schema_version": "0.3",
            "seed": 1,
            "num_agents": 2,
            "num_whales": 1,
            "conformity_threshold": 0.6,
            "whale_ratio": 0.2,
            "whale_signal": 1,
            "proposal": {"risk_change": 0.1, "yield_change": 0.2, "complexity": 0.3},
            "initial_votes": {"yes": 1, "no": 1, "abstain": 0},
            "final_votes": {"yes": 1, "no": 1, "abstain": 0},
            "weighted_initial": {"yes": 1.0, "no": 2.0, "abstain": 0.0},
            "weighted_final": {"yes": 1.0, "no": 2.0, "abstain": 0.0},
            "flips": 0,
            "agents": [
                {"is_whale": True, "risk_tolerance": 0.5, "greed": 0.5, "conformity": 0.5, "voting_power": 2.0, "initial_vote": 1, "final_vote": 1},
                {"is_whale": False, "risk_tolerance": 0.5, "greed": 0.5, "conformity": 0.5, "voting_power": 1.0, "initial_vote": -1, "final_vote": -1},
            ],
        }
        from migrate_results import migrate_record  # noqa: WPS433, E402

        migrated = migrate_record(legacy)
        self.assertIsNotNone(migrated)
        migrated = migrated or {}
        self.assertEqual(migrated["schema_version"], SCHEMA_VERSION)
        self.assertIn("cascade_rounds", migrated)
        self.assertIn("agents_flipped_total", migrated)
        self.assertIn("influence_model", migrated)

    @staticmethod
    def _canonicalize_result(result):
        """Strip non-deterministic fields and round floats for stable regression checks."""
        clean = dict(result)
        clean["run_id"] = "RUN_ID"
        clean["timestamp"] = "TIMESTAMP"

        def round_num(value):
            return round(value, 6) if isinstance(value, float) else value

        clean["proposal"] = {k: round_num(v) for k, v in clean["proposal"].items()}
        for block in ["initial_votes", "final_votes"]:
            clean[block] = dict(clean[block])
        for block in ["weighted_initial", "weighted_final"]:
            clean[block] = {k: round_num(v) for k, v in clean[block].items()}
        agents = []
        for agent in clean["agents"]:
            agents.append({k: round_num(v) if isinstance(v, float) else v for k, v in agent.items()})
        clean["agents"] = agents
        return clean

    @staticmethod
    def _load_fixture(path: Path):
        if not path.exists():
            raise AssertionError(f"Snapshot fixture missing at {path}")
        content = path.read_text(encoding="utf-8")
        return json.loads(content)


if __name__ == "__main__":
    unittest.main()
