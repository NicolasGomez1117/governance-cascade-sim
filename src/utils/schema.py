import uuid
from datetime import datetime
from typing import Any, Dict, List

SCHEMA_VERSION = "0.4"
ALLOWED_AGENT_TYPES = {
    "optimizer",
    "reputation_seeker",
    "contrarian",
    "institution",
    "retail",
    "whale",
}


def default_votes() -> Dict[str, float]:
    return {"yes": 0, "no": 0, "abstain": 0}


def is_uuid4(value: str) -> bool:
    try:
        uuid_obj = uuid.UUID(value, version=4)
    except (ValueError, TypeError):
        return False
    return str(uuid_obj) == value


def validate_result_schema(result: Dict[str, Any]) -> None:
    """
    Validate the shape and types of a simulation result.
    Raises ValueError on any schema mismatch.
    """
    required_keys = {
        "schema_version",
        "run_id",
        "timestamp",
        "seed",
        "num_agents",
        "num_whales",
        "whale_signal",
        "conformity_threshold",
        "whale_ratio",
        "proposal",
        "initial_votes",
        "final_votes",
        "weighted_initial",
        "weighted_final",
        "flips",
        "cascade_rounds",
        "agents_flipped_total",
        "influence_model",
        "agents",
    }
    present_keys = set(result.keys())
    missing = required_keys - present_keys
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    extra = present_keys - required_keys
    if extra:
        raise ValueError(f"Unexpected fields: {sorted(extra)}")

    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    if not isinstance(result["run_id"], str) or not is_uuid4(result["run_id"]):
        raise ValueError("run_id must be a UUID4 string")

    # Basic ISO timestamp format check.
    if not isinstance(result["timestamp"], str):
        raise ValueError("timestamp must be a string")
    try:
        datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601 format") from exc

    int_fields = ["seed", "num_agents", "num_whales", "whale_signal", "flips"]
    for field in int_fields:
        if not isinstance(result[field], int):
            raise ValueError(f"{field} must be int")

    float_fields = ["conformity_threshold", "whale_ratio"]
    for field in float_fields:
        if not isinstance(result[field], (int, float)):
            raise ValueError(f"{field} must be float")

    if not isinstance(result["cascade_rounds"], int) or result["cascade_rounds"] < 1:
        raise ValueError("cascade_rounds must be int >= 1")
    if not isinstance(result["agents_flipped_total"], int) or result["agents_flipped_total"] < 0:
        raise ValueError("agents_flipped_total must be int >= 0")
    if not isinstance(result["influence_model"], str):
        raise ValueError("influence_model must be a string")
    allowed_influence = {"erdos_renyi", "small_world", "none"}
    if result["influence_model"] not in allowed_influence:
        raise ValueError(f"influence_model must be one of {sorted(allowed_influence)}")

    _validate_vote_block(result["initial_votes"], field_name="initial_votes", expect_int=True)
    _validate_vote_block(result["final_votes"], field_name="final_votes", expect_int=True)
    _validate_vote_block(result["weighted_initial"], field_name="weighted_initial", expect_int=False)
    _validate_vote_block(result["weighted_final"], field_name="weighted_final", expect_int=False)

    proposal = result["proposal"]
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be a dict")
    proposal_keys = set(proposal.keys())
    expected_proposal = {"risk_change", "yield_change", "complexity"}
    missing_proposal = expected_proposal - proposal_keys
    extra_proposal = proposal_keys - expected_proposal
    if missing_proposal:
        raise ValueError(f"proposal missing {sorted(missing_proposal)}")
    if extra_proposal:
        raise ValueError(f"proposal has unexpected fields {sorted(extra_proposal)}")
    for key in expected_proposal:
        if not isinstance(proposal[key], (int, float)):
            raise ValueError(f"proposal.{key} must be numeric")

    agents = result["agents"]
    if not isinstance(agents, list):
        raise ValueError("agents must be a list")
    for idx, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ValueError(f"agents[{idx}] must be a dict")
        expected_agent_fields = {
            "is_whale",
            "risk_tolerance",
            "greed",
            "conformity",
            "voting_power",
            "initial_vote",
            "final_vote",
            "agent_type",
        }
        agent_keys = set(agent.keys())
        missing_agent_fields = expected_agent_fields - agent_keys
        if missing_agent_fields:
            raise ValueError(f"agents[{idx}] missing fields {sorted(missing_agent_fields)}")
        extra_agent_fields = agent_keys - expected_agent_fields
        if extra_agent_fields:
            raise ValueError(f"agents[{idx}] unexpected fields {sorted(extra_agent_fields)}")
        if not isinstance(agent["agent_type"], str):
            raise ValueError(f"agents[{idx}].agent_type must be str")
        if agent["agent_type"] not in ALLOWED_AGENT_TYPES:
            raise ValueError(f"agents[{idx}].agent_type must be one of {sorted(ALLOWED_AGENT_TYPES)}")
        if not isinstance(agent["is_whale"], bool):
            raise ValueError(f"agents[{idx}].is_whale must be bool")
        numeric_fields = ["risk_tolerance", "greed", "conformity", "voting_power", "initial_vote", "final_vote"]
        for field in numeric_fields:
            if not isinstance(agent[field], (int, float)):
                raise ValueError(f"agents[{idx}].{field} must be numeric")


def _validate_vote_block(block: Dict[str, Any], field_name: str, expect_int: bool) -> None:
    if not isinstance(block, dict):
        raise ValueError(f"{field_name} must be a dict")
    expected_keys = {"yes", "no", "abstain"}
    block_keys = set(block.keys())
    missing = expected_keys - block_keys
    if missing:
        raise ValueError(f"{field_name} missing {sorted(missing)}")
    extra = block_keys - expected_keys
    if extra:
        raise ValueError(f"{field_name} has unexpected fields {sorted(extra)}")
    for key in expected_keys:
        if expect_int and not isinstance(block[key], int):
            raise ValueError(f"{field_name}.{key} must be int")
        if not expect_int and not isinstance(block[key], (int, float)):
            raise ValueError(f"{field_name}.{key} must be numeric")
