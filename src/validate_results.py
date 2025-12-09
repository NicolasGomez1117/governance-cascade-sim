import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from utils.schema import validate_result_schema


def validate_file(path: Path) -> Tuple[int, List[str]]:
    """
    Validate each JSON object in a JSONL file against the result schema.
    Returns (records_checked, errors).
    """
    errors: List[str] = []
    if not path.exists():
        return 0, [f"File not found: {path}"]

    try:
        content = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return 0, [f"Failed to read {path}: {exc}"]

    checked = 0
    for idx, line in enumerate(content, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {idx}: invalid JSON ({exc})")
            continue
        try:
            validate_result_schema(record)
        except ValueError as exc:
            errors.append(f"Line {idx}: {exc}")
            continue
        checked += 1

    return checked, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate simulator JSONL outputs against the schema.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("results/results.jsonl"),
        help="Path to JSONL results file (default: results/results.jsonl)",
    )
    args = parser.parse_args()

    checked, errors = validate_file(args.file)
    if errors:
        print(f"Validation failed for {args.file}:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print(f"Validated {checked} record(s) in {args.file} with 0 errors.")


if __name__ == "__main__":
    main()
