import argparse
import json
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from influence import InfluenceNetwork
from utils.schema import ALLOWED_AGENT_TYPES, SCHEMA_VERSION, validate_result_schema


@dataclass
class Proposal:
    risk_change: float  # -1 (huge risk reduction) to +1 (huge risk increase)
    yield_change: float  # -1 (yield drop) to +1 (yield boost)
    complexity: float  # 0 (simple) to 1 (very complex)


@dataclass
class Agent:
    agent_type: str
    risk_tolerance: float  # 0 (risk averse) to 1 (risk seeking)
    greed: float  # 0 (yield agnostic) to 1 (yield maximizer)
    conformity: float  # 0 (independent) to 1 (follows whales)
    voting_power: float  # numeric weight; whales typically higher
    is_whale: bool

    def decide_vote(self, proposal: Proposal, whale_signal: Optional[int]) -> int:
        """
        Returns +1 for YES, -1 for NO, 0 for ABSTAIN.
        Decision rule mixes personal utility with archetype-specific adjustments and an optional whale signal.
        """
        personal_score = self._personal_utility(proposal)
        personal_score = self._apply_noise(personal_score)
        personal_score = self._apply_signal(personal_score, whale_signal)

        if personal_score > 0.05:
            return 1
        if personal_score < -0.05:
            return -1
        return 0

    def _personal_utility(self, proposal: Proposal) -> float:
        # Greed chases yield, risk_tolerance mutes risk aversion.
        yield_component = proposal.yield_change * self.greed
        risk_component = proposal.risk_change * (1 - self.risk_tolerance)
        complexity_penalty = proposal.complexity * (1 - self.risk_tolerance) * 0.4
        base = yield_component - risk_component - complexity_penalty

        # Archetype-specific bias toward/against complexity and risk.
        if self.agent_type == "institution":
            base -= proposal.complexity * 0.1
        elif self.agent_type == "retail":
            base += proposal.yield_change * 0.05
        elif self.agent_type == "contrarian":
            base -= proposal.risk_change * 0.05
        return base

    def _noise_amplitude(self) -> float:
        noise_by_type = {
            "optimizer": 0.05,
            "reputation_seeker": 0.05,
            "contrarian": 0.03,
            "institution": 0.01,
            "retail": 0.12,
            "whale": 0.02,
        }
        return noise_by_type.get(self.agent_type, 0.05)

    def _apply_noise(self, score: float) -> float:
        noise = random.uniform(-self._noise_amplitude(), self._noise_amplitude())
        return score + noise

    def _apply_signal(self, score: float, whale_signal: Optional[int]) -> float:
        if whale_signal is None or whale_signal == 0:
            return score

        magnitude = max(abs(score), 0.1)
        if self.agent_type == "contrarian":
            sway = -self.conformity * whale_signal * magnitude
            return 0.6 * score + 0.4 * sway
        if self.agent_type == "reputation_seeker":
            sway = 1.2 * self.conformity * whale_signal * magnitude
            return 0.55 * score + 0.45 * sway
        if self.agent_type == "institution":
            sway = 0.4 * self.conformity * whale_signal * magnitude * 0.5
            return 0.8 * score + 0.2 * sway
        if self.agent_type == "retail":
            sway = 0.8 * self.conformity * whale_signal * magnitude
            return 0.65 * score + 0.35 * sway

        # Optimizer/whale defaults.
        sway = 0.6 * self.conformity * whale_signal * magnitude
        return 0.7 * score + 0.3 * sway


def generate_agents(n: int, whale_ratio: float = 0.15) -> List[Agent]:
    agents: List[Agent] = []
    for _ in range(n):
        is_whale = random.random() < whale_ratio
        agent_type = "whale" if is_whale else _sample_agent_type()
        agent = _build_agent(agent_type, is_whale)
        agents.append(agent)
    return agents


def _sample_agent_type() -> str:
    """
    Deterministic-friendly sampler over non-whale archetypes.
    Weights favor optimizers and reputation seekers.
    """
    choices: List[Tuple[str, float]] = [
        ("optimizer", 0.30),
        ("reputation_seeker", 0.20),
        ("contrarian", 0.15),
        ("institution", 0.15),
        ("retail", 0.20),
    ]
    roll = random.random()
    cumulative = 0.0
    for archetype, weight in choices:
        cumulative += weight
        if roll <= cumulative:
            return archetype
    return choices[-1][0]


def _build_agent(agent_type: str, is_whale: bool) -> Agent:
    if agent_type not in ALLOWED_AGENT_TYPES:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    if agent_type == "whale":
        return Agent(
            agent_type="whale",
            risk_tolerance=random.uniform(0.2, 0.8),
            greed=random.uniform(0.4, 1.0),
            conformity=random.uniform(0.1, 0.6),
            voting_power=random.uniform(2.0, 5.0),
            is_whale=True,
        )
    if agent_type == "optimizer":
        return Agent(
            agent_type=agent_type,
            risk_tolerance=random.uniform(0.2, 0.9),
            greed=random.uniform(0.5, 1.0),
            conformity=random.uniform(0.2, 0.8),
            voting_power=random.uniform(0.3, 1.2),
            is_whale=is_whale,
        )
    if agent_type == "reputation_seeker":
        return Agent(
            agent_type=agent_type,
            risk_tolerance=random.uniform(0.3, 0.7),
            greed=random.uniform(0.2, 0.6),
            conformity=random.uniform(0.6, 1.0),
            voting_power=random.uniform(0.2, 1.0),
            is_whale=is_whale,
        )
    if agent_type == "contrarian":
        return Agent(
            agent_type=agent_type,
            risk_tolerance=random.uniform(0.4, 0.9),
            greed=random.uniform(0.4, 0.8),
            conformity=random.uniform(0.1, 0.6),
            voting_power=random.uniform(0.2, 1.0),
            is_whale=is_whale,
        )
    if agent_type == "institution":
        return Agent(
            agent_type=agent_type,
            risk_tolerance=random.uniform(0.1, 0.5),
            greed=random.uniform(0.3, 0.7),
            conformity=random.uniform(0.2, 0.6),
            voting_power=random.uniform(0.8, 1.5),
            is_whale=is_whale,
        )
    # retail
    return Agent(
        agent_type=agent_type,
        risk_tolerance=random.uniform(0.3, 0.8),
        greed=random.uniform(0.2, 1.0),
        conformity=random.uniform(0.3, 0.9),
        voting_power=random.uniform(0.1, 0.9),
        is_whale=is_whale,
    )


def tally_votes(votes: List[int]) -> Tuple[int, int, int]:
    yes = votes.count(1)
    no = votes.count(-1)
    abstain = votes.count(0)
    return yes, no, abstain


def weighted_results(agents: List[Agent], votes: List[int]) -> Tuple[float, float, float]:
    yes_weight = sum(a.voting_power for a, v in zip(agents, votes) if v == 1)
    no_weight = sum(a.voting_power for a, v in zip(agents, votes) if v == -1)
    abstain_weight = sum(a.voting_power for a, v in zip(agents, votes) if v == 0)
    return yes_weight, no_weight, abstain_weight


def whale_majority_signal(agents: List[Agent], votes: List[int]) -> int:
    whale_votes = [v for a, v in zip(agents, votes) if a.is_whale]
    yes, no, _ = tally_votes(whale_votes)
    if yes > no:
        return 1
    if no > yes:
        return -1
    return 0


def multi_round_cascade(
    agents: List[Agent],
    proposal: Proposal,
    initial_votes: List[int],
    network: InfluenceNetwork,
    conformity_threshold: float,
    max_rounds: int = 5,
) -> Tuple[List[int], int, int]:
    """
    Propagate influence over the network until convergence or max rounds.
    Returns final_votes, cascade_rounds, total_flips.
    """
    current_votes = list(initial_votes)
    total_flips = 0
    rounds = 0

    for _ in range(max_rounds):
        rounds += 1
        flipped_this_round = 0
        next_votes: List[int] = list(current_votes)

        for idx, agent in enumerate(agents):
            neighbor_score = network.influence_score(idx, current_votes)
            if agent.conformity < conformity_threshold or abs(neighbor_score) < conformity_threshold or neighbor_score == 0:
                continue
            neighbor_signal = 1 if neighbor_score > 0 else -1
            new_vote = agent.decide_vote(proposal, neighbor_signal)
            if new_vote != current_votes[idx]:
                next_votes[idx] = new_vote
                flipped_this_round += 1

        current_votes = next_votes
        total_flips += flipped_this_round
        if flipped_this_round == 0:
            break

    return current_votes, rounds, total_flips


def simulate_once(
    num_agents: int,
    conformity_threshold: float,
    whale_ratio: float,
    seed: int,
    influence_model: str,
    network_density: float,
    rewire_prob: float,
) -> Dict[str, object]:
    random.seed(seed)
    network = InfluenceNetwork(
        num_agents=num_agents,
        model=influence_model,
        density=network_density,
        rewire_prob=rewire_prob,
        seed=seed,
    )
    proposal = Proposal(
        risk_change=random.uniform(-0.5, 0.7),
        yield_change=random.uniform(-0.2, 0.9),
        complexity=random.uniform(0.1, 0.8),
    )
    agents = generate_agents(num_agents, whale_ratio=whale_ratio)

    # Round 0: initial personal votes (no social signal yet).
    initial_votes: List[int] = [agent.decide_vote(proposal, whale_signal=None) for agent in agents]

    final_votes, cascade_rounds, total_flips = multi_round_cascade(
        agents,
        proposal,
        initial_votes,
        network,
        conformity_threshold=conformity_threshold,
        max_rounds=5,
    )

    initial_counts = tally_votes(initial_votes)
    final_counts = tally_votes(final_votes)
    weighted_initial = weighted_results(agents, initial_votes)
    weighted_final = weighted_results(agents, final_votes)

    # Capture per-agent details for downstream analysis.
    agent_records = []
    for agent, init_vote, final_vote in zip(agents, initial_votes, final_votes):
        agent_records.append(
            {
                "agent_type": agent.agent_type,
                "is_whale": agent.is_whale,
                "risk_tolerance": agent.risk_tolerance,
                "greed": agent.greed,
                "conformity": agent.conformity,
                "voting_power": agent.voting_power,
                "initial_vote": init_vote,
                "final_vote": final_vote,
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "num_agents": num_agents,
        "num_whales": sum(1 for a in agents if a.is_whale),
        "conformity_threshold": conformity_threshold,
        "whale_ratio": whale_ratio,
        "whale_signal": whale_majority_signal(agents, initial_votes),
        "influence_model": network.model,
        "proposal": asdict(proposal),
        "initial_votes": {"yes": initial_counts[0], "no": initial_counts[1], "abstain": initial_counts[2]},
        "final_votes": {"yes": final_counts[0], "no": final_counts[1], "abstain": final_counts[2]},
        "weighted_initial": {"yes": weighted_initial[0], "no": weighted_initial[1], "abstain": weighted_initial[2]},
        "weighted_final": {"yes": weighted_final[0], "no": weighted_final[1], "abstain": weighted_final[2]},
        "flips": total_flips,
        "cascade_rounds": cascade_rounds,
        "agents_flipped_total": total_flips,
        "agents": agent_records,
    }
    validate_result_schema(result)
    return result


def write_result_jsonl(path: str, data: Dict[str, object]) -> None:
    """Append a single JSON object to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def run_simulation(
    num_agents: int = 50,
    conformity_threshold: float = 0.6,
    whale_ratio: float = 0.15,
    runs: int = 1,
    seed: int = 42,
    output: Optional[str] = None,
    influence_model: str = "erdos_renyi",
    network_density: float = 0.2,
    rewire_prob: float = 0.1,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for run_idx in range(runs):
        run_seed = seed + run_idx
        result = simulate_once(
            num_agents,
            conformity_threshold,
            whale_ratio,
            run_seed,
            influence_model=influence_model,
            network_density=network_density,
            rewire_prob=rewire_prob,
        )
        results.append(result)
        print_run(result, run_idx + 1, runs)
        if output:
            write_result_jsonl(output, result)

    if output:
        print(f"\nAppended {len(results)} run(s) to {output}")

    return results


def print_run(result: Dict[str, object], run_number: int, total_runs: int) -> None:
    proposal = result["proposal"]
    initial = result["initial_votes"]
    final = result["final_votes"]
    weighted_initial = result["weighted_initial"]
    weighted_final = result["weighted_final"]
    whale_signal = result.get("whale_signal", 0)

    print(f"\n=== Run {run_number}/{total_runs} ({result.get('run_id', 'unknown')}) ===")
    print(
        "Proposal: "
        f"risk_change={proposal['risk_change']:.2f}, "
        f"yield_change={proposal['yield_change']:.2f}, "
        f"complexity={proposal['complexity']:.2f}"
    )
    print(f"Agents: {result.get('num_agents', len(result.get('agents', [])))} total, {result.get('num_whales', 0)} whales")
    print(f"Influence model: {result.get('influence_model', 'unknown')} | Cascade rounds: {result.get('cascade_rounds', '?')}")
    print("Round 1 (before cascade):")
    print(f"  YES={initial['yes']}, NO={initial['no']}, ABSTAIN={initial['abstain']}")
    print(
        "  Weighted "
        f"YES={weighted_initial['yes']:.2f}, "
        f"NO={weighted_initial['no']:.2f}, "
        f"ABSTAIN={weighted_initial['abstain']:.2f}"
    )
    print(f"  Whale majority signal: {whale_signal:+d}")
    print("After cascade:")
    print(f"  YES={final['yes']}, NO={final['no']}, ABSTAIN={final['abstain']}")
    print(
        "  Weighted "
        f"YES={weighted_final['yes']:.2f}, "
        f"NO={weighted_final['no']:.2f}, "
        f"ABSTAIN={weighted_final['abstain']:.2f}"
    )
    print(f"Agents that flipped during cascade: {result['flips']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Governance Voting Cascade Simulator v0.2")
    parser.add_argument("--agents", type=int, default=50, help="Number of agents to simulate (default: 50)")
    parser.add_argument(
        "--conformity-threshold",
        type=float,
        default=0.6,
        help="Minimum conformity to follow whale signal in round 2 (default: 0.6)",
    )
    parser.add_argument("--whale-ratio", type=float, default=0.15, help="Fraction of agents that are whales (default: 0.15)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (default: 42)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSONL results",
    )
    parser.add_argument(
        "--influence-model",
        type=str,
        choices=["erdos_renyi", "small_world"],
        default="erdos_renyi",
        help="Influence network model (default: erdos_renyi)",
    )
    parser.add_argument(
        "--network-density",
        type=float,
        default=0.2,
        help="Edge probability/density for influence graph (default: 0.2)",
    )
    parser.add_argument(
        "--rewire-prob",
        type=float,
        default=0.1,
        help="Rewire probability for small_world model (default: 0.1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        num_agents=args.agents,
        conformity_threshold=args.conformity_threshold,
        whale_ratio=args.whale_ratio,
        runs=args.runs,
        seed=args.seed,
        output=args.output,
        influence_model=args.influence_model,
        network_density=args.network_density,
        rewire_prob=args.rewire_prob,
    )
