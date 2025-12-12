# Governance Cascade Simulator

Governance Cascade Simulator is a hybrid project: part practical engineering lab for cloud/DevOps/ML-infra/Kubernetes skills, part exploratory research tool for studying governance behavior, whale influence, and cascade effects in decentralized systems. The goal is to showcase infrastructure discipline while enabling agent-based economic modeling.

## What It Does Today (v0.4)
- Proposal and Agent dataclasses with risk/yield/complexity modeling
- Multiple agent archetypes (optimizer, reputation seeker, contrarian, institutional, retail, whale) with differentiated decision logic
- Network-based influence propagation (Erdos–Renyi or small-world) with multi-round cascades
- Conformity-aware decision rule with whale-majority signaling and cascade tracking
- Two-round cascade (initial vote, influence phase)
- Weighted voting results and tracking of vote flips
- Per-agent records exported for downstream analysis (agent attributes + votes)
- JSONL logging with append mode for incremental runs
- Detailed agent-level data capture for statistical analysis
- Versioned outputs (`schema_version: "0.4"`) with run metadata (`run_id`, ISO timestamp, seed, params)

## Running the Simulator
Examples:
- `python3 src/simulator.py --agents 100 --runs 20 --seed 42 --output results/results.jsonl`
- `python3 src/simulator.py --agents 50 --conformity-threshold 0.7 --whale-ratio 0.2 --output results/results.jsonl`

Flags:
- `--agents`: number of agents (default: 50)
- `--runs`: number of simulation runs (default: 1)
- `--seed`: base RNG seed (default: 42)
- `--conformity-threshold`: minimum conformity to follow whale signal (default: 0.6)
- `--whale-ratio`: fraction of agents that are whales (default: 0.15)
- `--influence-model`: influence network type (`erdos_renyi` default, `small_world` optional)
- `--network-density`: edge probability/density for the influence graph (default: 0.2)
- `--rewire-prob`: rewiring probability for small-world graphs (default: 0.1)
- `--output`: optional JSONL output path for run summaries

## Schema Discipline
- Each run emits a uniform schema with `schema_version: "0.4"`, `run_id` (UUID4), and ISO timestamp.
- Consistent keys across runs: proposal block, votes (initial/final/weighted), flips, per-agent records, and run parameters.
- Influence metadata: `influence_model`, `cascade_rounds`, and `agents_flipped_total` capture cascade dynamics.
- Use `migrate_results.py <old.jsonl> <new.jsonl>` to normalize legacy outputs into the v0.4 schema.
- Determinism: fixed seeds produce identical agent generation, influence networks, cascade rounds, and flip counts for regression stability.

## Project Layout
```
.
├── src/
│   ├── simulator.py
│   └── utils/
│       └── schema.py
├── results/          # default location for JSONL outputs
├── tests/
│   └── test_schema.py
├── migrate_results.py
└── README.md
```

## Tests
- Minimal schema check: `python -m unittest tests/test_schema.py`
- Validate outputs: `python src/validate_results.py --file results/results.jsonl`
- CI: GitHub Actions runs unit tests, generates sample outputs, validates them, and only uploads validated artifacts.

## Engineering Roadmap
- Containerization with Docker
- Push images to ECR and deploy on EKS
- Terraform-managed cluster for reproducibility
- Prometheus/Grafana metrics collection
- Autoscaling parameter sweeps via Kubernetes Jobs
- Infrastructure as Code guardrails and CI/CD

## Research Roadmap
- Multi-dimensional proposal vectors
- Rich agent heterogeneity
- Whale coalition modeling
- Influence networks and cascading behavior
- Governance attack modeling
- MEV-aware simulations
- Agent trace analysis
- Statistical analysis and visualization

## Why This Project Matters for Hiring
- Demonstrates DevOps, cloud, infrastructure, and simulation-engineering skills through containerization, orchestration, and IaC plans.
- Highlights research thinking via agent-based modeling, governance dynamics, and MEV/influence analysis.
- Designed to be both internship-ready (clear runs, logging, reproducible seeds) and long-term research relevant (roadmaps for deeper modeling and infra scale-out).

## Suggested Folder Structure (future)
```
.
├── simulator.py
├── README.md
├── infra/
│   ├── docker/
│   ├── terraform/
│   └── k8s/
├── data/
│   └── runs/
├── notebooks/
├── dashboards/
└── scripts/
```
