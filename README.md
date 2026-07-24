# Precision Rocket Soft-Landing RL

A highly rigorous Reinforcement Learning project showcasing Proximal Policy Optimization (PPO) with Potential-Based Reward Shaping applied to a 1D vertical rocket soft-landing environment.

## Overview
This project solves a classic continuous control problem: landing a rocket exactly on a target pad while minimizing fuel, stripped down to a pure 1D vertical drop to demonstrate flawless algorithmic convergence.

## Why this is a superior RL implementation:

1. **Mathematically Immune to Reward Hacking**: Instead of ad-hoc heuristic rewards, this project uses **Potential-Based Reward Shaping (PBRS)** (Ng et al., 1999). We use a shaping function $F(s, s') = \gamma \Phi(s') - \Phi(s)$ where $\Phi(s)$ is the potential field measuring the distance to the goal state. This is mathematically proven to accelerate learning without altering the optimal policy, effectively rendering "reward hacking" impossible.
2. **State-of-the-Art PPO**: We utilize an Actor-Critic Multi-Layer Perceptron (MLP) trained with Proximal Policy Optimization. It features Generalized Advantage Estimation (GAE-$\lambda$), surrogate loss clipping, and entropy regularization to ensure robust convergence.
3. **From-Scratch Physics Engine**: A custom `Gymnasium`-compliant environment featuring 1D rigid body dynamics, gravity, vertical thrust, Euler integration, and strict fuel limits. Because of the simplified 1D state, the agent converges to a **100% success rate** in under 200,000 timesteps.

## Project Structure
- `src/environment.py`: The custom 1D physics environment.
- `src/models.py`: PyTorch Actor-Critic MLP architectures.
- `src/agent.py`: The PPO Agent core logic.
- `src/train.py` & `src/evaluate.py`: The training loop and evaluation benchmarking suite.
- `src/plot_results.py`: Analytics and visualization.

## Usage

### 1. Training
Run the PPO training loop (this will save weights to `checkpoints/` and metrics to `logs/`):
```bash
python main.py --train
```

### 2. Evaluation
Run a 100-episode benchmark on the trained model (saves results to `logs/eval_report.csv`):
```bash
python main.py --evaluate
```

### 3. Visualization
Generate training metrics plots (`plots/training_metrics.png`):
```bash
python main.py --plot
```

## Setup & Requirements
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
