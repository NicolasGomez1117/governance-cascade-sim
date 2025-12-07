# Governance Cascade Simulator

Governance Cascade Simulator is a hybrid project: part practical engineering lab for cloud/DevOps/ML-infra/Kubernetes skills, part exploratory research tool for studying governance behavior, whale influence, and cascade effects in decentralized systems. The goal is to showcase infrastructure discipline while enabling agent-based economic modeling.

## What It Does Today (v0.2)
- Proposal and Agent dataclasses with risk/yield/complexity modeling
- Conformity-aware decision rule with whale-majority signaling
- Two-round cascade (initial vote, influence phase)
- Weighted voting results and tracking of vote flips
- Per-agent records exported for downstream analysis (agent attributes + votes)
- JSONL logging with append mode for incremental runs
- Detailed agent-level data capture for statistical analysis

## Running the Simulator
Examples:
- `python simulator.py --agents 100 --runs 20 --seed 42`
- `python simulator.py --agents 50 --conformity-threshold 0.7 --output results.jsonl`

Flags:
- `--agents`: number of agents (default: 50)
- `--runs`: number of simulation runs (default: 1)
- `--seed`: base RNG seed (default: 42)
- `--conformity-threshold`: minimum conformity to follow whale signal (default: 0.6)
- `--whale-ratio`: fraction of agents that are whales (default: 0.15)
- `--output`: optional JSONL output path for run summaries

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
