import random
from typing import Dict, Iterable, List, Tuple


class InfluenceNetwork:
    """
    Deterministic influence graph for agent interactions.
    influence[i][j] = weight of agent i influencing agent j.
    """

    def __init__(
        self,
        num_agents: int,
        model: str = "erdos_renyi",
        density: float = 0.2,
        rewire_prob: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.num_agents = num_agents
        self.model = model
        self.density = max(0.0, min(1.0, density))
        self.rewire_prob = max(0.0, min(1.0, rewire_prob))
        self._rng = random.Random(seed)
        self.influence: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_agents)}
        if model == "small_world":
            self._build_small_world()
        else:
            self.model = "erdos_renyi"
            self._build_erdos_renyi()

    def neighbors(self, agent_id: int) -> Iterable[Tuple[int, float]]:
        return self.influence.get(agent_id, [])

    def influence_score(self, agent_id: int, votes: List[int]) -> float:
        """
        Weighted average of neighbor votes; returns 0 if no incoming edges.
        """
        weighted_sum = 0.0
        total = 0.0
        for neighbor_id, weight in self.neighbors(agent_id):
            weighted_sum += weight * votes[neighbor_id]
            total += weight
        if total == 0:
            return 0.0
        return weighted_sum / total

    def _build_erdos_renyi(self) -> None:
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    continue
                if self._rng.random() <= self.density:
                    weight = self._rng.uniform(0.1, 1.0)
                    # Edge: i influences j, store as incoming for j.
                    self.influence[j].append((i, weight))

    def _build_small_world(self) -> None:
        # Start with a ring lattice then rewire edges with probability.
        k = max(1, int(self.density * self.num_agents))
        if k % 2 != 0:
            k += 1  # ensure even neighbors for symmetric lattice
        half_k = max(1, k // 2)

        for i in range(self.num_agents):
            for offset in range(1, half_k + 1):
                j = (i + offset) % self.num_agents
                weight = self._rng.uniform(0.1, 1.0)
                self.influence[j].append((i, weight))

        # Rewire directed edges with probability rewire_prob.
        for target in range(self.num_agents):
            new_edges: List[Tuple[int, float]] = []
            for source, weight in self.influence[target]:
                if self._rng.random() <= self.rewire_prob:
                    candidate = self._rng.randrange(self.num_agents)
                    while candidate == target:
                        candidate = self._rng.randrange(self.num_agents)
                    new_edges.append((candidate, self._rng.uniform(0.1, 1.0)))
                else:
                    new_edges.append((source, weight))
            self.influence[target] = new_edges
