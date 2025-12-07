import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from simulator import run_simulation  # noqa: E402
from utils.schema import SCHEMA_VERSION, validate_result_schema  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
