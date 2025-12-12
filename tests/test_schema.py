import json
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
        fixture_path = Path(__file__).parent / "fixtures" / "snapshot_v0_2.json"
        result = run_simulation(num_agents=5, runs=1, seed=11, conformity_threshold=0.6, whale_ratio=0.2, output=None)[0]
        canonical = self._canonicalize_result(result)
        fixture = self._load_fixture(fixture_path)
        self.assertEqual(canonical, fixture)

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
            agents.append({k: round_num(v) if k != "is_whale" else v for k, v in agent.items()})
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
