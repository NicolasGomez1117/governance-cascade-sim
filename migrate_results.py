"""
Migration utility to normalize legacy JSONL outputs to schema version 0.4.
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure local imports work when running as a script.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.schema import ALLOWED_AGENT_TYPES, SCHEMA_VERSION, validate_result_schema  # noqa: E402


def migrate_record(raw: Dict[str, Any], default_seed: int = 0) -> Optional[Dict[str, Any]]:
    """
    Transform a legacy record into the v0.4 schema. Returns None if migration fails.
    """
    def vote_block(source: Dict[str, Any], keys) -> Optional[Dict[str, Any]]:
        for key in keys:
            if key in source and isinstance(source[key], dict):
                block = source[key]
                if {"yes", "no", "abstain"} <= set(block.keys()):
                    return {
                        "yes": block.get("yes", 0),
                        "no": block.get("no", 0),
                        "abstain": block.get("abstain", 0),
                    }
        return None

    proposal = raw.get("proposal", {})
    proposal_block = {
        "risk_change": proposal.get("risk_change", 0.0),
        "yield_change": proposal.get("yield_change", 0.0),
        "complexity": proposal.get("complexity", 0.0),
    }

    agents = raw.get("agents", [])
    if not isinstance(agents, list):
        print("Warning: agents not list; dropping agents", file=sys.stderr)
        agents = []

    # Normalize counts/weights blocks.
    initial_votes = vote_block(raw, ["initial_votes", "initial_counts"]) or {"yes": 0, "no": 0, "abstain": 0}
    final_votes = vote_block(raw, ["final_votes", "final_counts"]) or {"yes": 0, "no": 0, "abstain": 0}
    weighted_initial = vote_block(raw, ["weighted_initial"]) or {"yes": 0.0, "no": 0.0, "abstain": 0.0}
    weighted_final = vote_block(raw, ["weighted_final"]) or {"yes": 0.0, "no": 0.0, "abstain": 0.0}

    num_agents = raw.get("num_agents", len(agents))
    num_whales = raw.get("num_whales", sum(1 for a in agents if isinstance(a, dict) and a.get("is_whale")))

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": int(raw.get("seed", default_seed)),
        "num_agents": int(num_agents) if isinstance(num_agents, (int, float)) else 0,
        "num_whales": int(num_whales) if isinstance(num_whales, (int, float)) else 0,
        "conformity_threshold": float(raw.get("conformity_threshold", 0.6)),
        "whale_ratio": float(raw.get("whale_ratio", 0.15)),
        "whale_signal": int(raw.get("whale_signal", 0)),
        "proposal": proposal_block,
        "initial_votes": initial_votes,
        "final_votes": final_votes,
        "weighted_initial": weighted_initial,
        "weighted_final": weighted_final,
        "flips": int(raw.get("flips", 0)),
        "cascade_rounds": max(1, int(raw.get("cascade_rounds", 1))),
        "agents_flipped_total": max(0, int(raw.get("agents_flipped_total", raw.get("flips", 0)))),
        "influence_model": raw.get("influence_model", "none"),
        "agents": normalize_agents(agents),
    }

    try:
        validate_result_schema(result)
    except ValueError as exc:
        print(f"Warning: dropping record due to schema validation error: {exc}", file=sys.stderr)
        return None
    return result


def normalize_agents(agents: Any):
    normalized = []
    for agent in agents:
        if not isinstance(agent, dict):
            print("Warning: skipping non-dict agent", file=sys.stderr)
            continue
        is_whale = bool(agent.get("is_whale", False))
        agent_type = agent.get("agent_type")
        if agent_type not in ALLOWED_AGENT_TYPES:
            agent_type = "whale" if is_whale else "optimizer"
        normalized.append(
            {
                "agent_type": agent_type,
                "is_whale": is_whale,
                "risk_tolerance": float(agent.get("risk_tolerance", 0.0)),
                "greed": float(agent.get("greed", 0.0)),
                "conformity": float(agent.get("conformity", 0.0)),
                "voting_power": float(agent.get("voting_power", 0.0)),
                "initial_vote": int(agent.get("initial_vote", 0)),
                "final_vote": int(agent.get("final_vote", 0)),
            }
        )
    return normalized


def migrate_file(input_path: Path, output_path: Path) -> None:
    migrated = 0
    dropped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON on line {line_num}", file=sys.stderr)
                dropped += 1
                continue

            migrated_record = migrate_record(raw, default_seed=line_num)
            if migrated_record is None:
                dropped += 1
                continue

            outfile.write(json.dumps(migrated_record) + "\n")
            migrated += 1

    print(f"Migrated {migrated} record(s); dropped {dropped} record(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate JSONL results to schema version 0.4")
    parser.add_argument("input", type=Path, help="Path to legacy JSONL file")
    parser.add_argument("output", type=Path, help="Path to write migrated JSONL file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    migrate_file(args.input, args.output)
