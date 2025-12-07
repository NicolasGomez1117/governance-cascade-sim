import argparse
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class Proposal:
    risk_change: float  # -1 (huge risk reduction) to +1 (huge risk increase)
    yield_change: float  # -1 (yield drop) to +1 (yield boost)
    complexity: float  # 0 (simple) to 1 (very complex)


@dataclass
class Agent:
    risk_tolerance: float  # 0 (risk averse) to 1 (risk seeking)
    greed: float  # 0 (yield agnostic) to 1 (yield maximizer)
    conformity: float  # 0 (independent) to 1 (follows whales)
    voting_power: float  # numeric weight; whales typically higher
    is_whale: bool

    def decide_vote(self, proposal: Proposal, whale_signal: Optional[int]) -> int:
        """
        Returns +1 for YES, -1 for NO, 0 for ABSTAIN.
        Decision rule mixes personal utility with an optional whale signal.
        """
        # Personal utility: greed chases yield, risk_tolerance mutes risk aversion.
        yield_component = proposal.yield_change * self.greed
        risk_component = proposal.risk_change * (1 - self.risk_tolerance)
        complexity_penalty = proposal.complexity * (1 - self.risk_tolerance) * 0.4
        personal_score = yield_component - risk_component - complexity_penalty

        # Mild randomness to avoid perfectly rigid decisions.
        noise = random.uniform(-0.05, 0.05)
        personal_score += noise

        # Conformity pulls the vote toward the whale majority signal.
        if whale_signal is not None and whale_signal != 0:
            sway = self.conformity * whale_signal * max(abs(personal_score), 0.1)
            personal_score = 0.7 * personal_score + 0.3 * sway

        if personal_score > 0.05:
            return 1
        if personal_score < -0.05:
            return -1
        return 0


def generate_agents(n: int, whale_ratio: float = 0.15) -> List[Agent]:
    agents: List[Agent] = []
    for _ in range(n):
        is_whale = random.random() < whale_ratio
        agent = Agent(
            risk_tolerance=random.uniform(0.1, 0.9),
            greed=random.uniform(0.2, 1.0),
            conformity=random.uniform(0.0, 1.0),
            voting_power=random.uniform(2.0, 5.0) if is_whale else random.uniform(0.2, 1.0),
            is_whale=is_whale,
        )
        agents.append(agent)
    return agents


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


def simulate_once(num_agents: int, conformity_threshold: float, whale_ratio: float, seed: int) -> Dict[str, object]:
    random.seed(seed)
    proposal = Proposal(
        risk_change=random.uniform(-0.5, 0.7),
        yield_change=random.uniform(-0.2, 0.9),
        complexity=random.uniform(0.1, 0.8),
    )
    agents = generate_agents(num_agents, whale_ratio=whale_ratio)

    # Round 1: whales first, others vote without a whale signal.
    initial_votes: List[int] = [0] * len(agents)
    for idx, agent in enumerate(agents):
        if agent.is_whale:
            initial_votes[idx] = agent.decide_vote(proposal, whale_signal=None)
    for idx, agent in enumerate(agents):
        if not agent.is_whale:
            initial_votes[idx] = agent.decide_vote(proposal, whale_signal=None)

    whale_signal = whale_majority_signal(agents, initial_votes)

    # Round 2: conformist agents may shift toward the whale majority.
    final_votes: List[int] = []
    flips = 0
    for agent, vote in zip(agents, initial_votes):
        if agent.conformity >= conformity_threshold and whale_signal != 0:
            new_vote = agent.decide_vote(proposal, whale_signal)
            if new_vote != vote:
                flips += 1
            final_votes.append(new_vote)
        else:
            final_votes.append(vote)

    initial_counts = tally_votes(initial_votes)
    final_counts = tally_votes(final_votes)
    weighted_initial = weighted_results(agents, initial_votes)
    weighted_final = weighted_results(agents, final_votes)

    # Capture per-agent details for downstream analysis.
    agent_records = []
    for agent, init_vote, final_vote in zip(agents, initial_votes, final_votes):
        agent_records.append(
            {
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
        "seed": seed,
        "num_agents": num_agents,
        "num_whales": sum(1 for a in agents if a.is_whale),
        "whale_signal": whale_signal,
        "proposal": asdict(proposal),
        "initial_votes": {"yes": initial_counts[0], "no": initial_counts[1], "abstain": initial_counts[2]},
        "final_votes": {"yes": final_counts[0], "no": final_counts[1], "abstain": final_counts[2]},
        "weighted_initial": {"yes": weighted_initial[0], "no": weighted_initial[1], "abstain": weighted_initial[2]},
        "weighted_final": {"yes": weighted_final[0], "no": weighted_final[1], "abstain": weighted_final[2]},
        "flips": flips,
        "agents": agent_records,
    }
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
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for run_idx in range(runs):
        run_seed = seed + run_idx
        result = simulate_once(num_agents, conformity_threshold, whale_ratio, run_seed)
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

    print(f"\n=== Run {run_number}/{total_runs} ===")
    print(
        "Proposal: "
        f"risk_change={proposal['risk_change']:.2f}, "
        f"yield_change={proposal['yield_change']:.2f}, "
        f"complexity={proposal['complexity']:.2f}"
    )
    print(f"Agents: {result.get('num_agents', len(result.get('agents', [])))} total, {result.get('num_whales', 0)} whales")
    print("Round 1 (before cascade):")
    print(f"  YES={initial['yes']}, NO={initial['no']}, ABSTAIN={initial['abstain']}")
    print(
        "  Weighted "
        f"YES={weighted_initial['yes']:.2f}, "
        f"NO={weighted_initial['no']:.2f}, "
        f"ABSTAIN={weighted_initial['abstain']:.2f}"
    )
    print(f"  Whale majority signal: {whale_signal:+d}")
    print("Round 2 (after cascade):")
    print(f"  YES={final['yes']}, NO={final['no']}, ABSTAIN={final['abstain']}")
    print(
        "  Weighted "
        f"YES={weighted_final['yes']:.2f}, "
        f"NO={weighted_final['no']:.2f}, "
        f"ABSTAIN={weighted_final['abstain']:.2f}"
    )
    print(f"Agents that flipped after whale signal: {result['flips']}")


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
    )
