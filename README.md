# Smart Data Center Cooling Optimization using Reinforcement Learning

## Overview

Modern data centers consume significant energy for cooling infrastructure.  
Traditional cooling systems rely on static thresholds and rule-based control, leading to inefficient energy usage and unnecessary operational costs under dynamic workloads.

This project presents an **AI-driven cooling optimization system** that leverages **Reinforcement Learning (RL)** to dynamically regulate cooling levels based on real-time workload and temperature conditions.  
The system is designed with a production-oriented mindset, integrating **DevOps and MLOps practices** for deployment, automation, and monitoring.

---

## Problem Statement

Data center cooling systems must maintain safe operating temperatures while minimizing energy consumption. Static cooling strategies fail to adapt to fluctuating server workloads, resulting in:

- Excessive energy consumption  
- Inefficient cooling allocation  
- Lack of automated optimization  

The challenge is to build an intelligent system capable of **learning optimal cooling policies automatically** while ensuring thermal safety and operational stability.

---

## Our Solution

We implement a **Reinforcement Learning–based optimization agent** that continuously interacts with a simulated data center environment.

The system workflow:

1. A data center simulator generates workload and temperature states.
2. The RL agent observes system conditions.
3. The agent selects an optimal cooling action.
4. Cooling decisions are applied dynamically.
5. System metrics are monitored and deployed through an automated DevOps pipeline.

Key capabilities:

- Adaptive cooling control using RL
- Energy-efficient optimization strategy
- Modular architecture for easy integration
- Containerized deployment with monitoring support

---

## Tech Stack

### Reinforcement Learning
- Python  
- Gymnasium  
- Stable-Baselines3 (PPO)  
- PyTorch  

### Backend & Simulation
- FastAPI  
- Uvicorn  
- NumPy  
- Pydantic  

### DevOps & MLOps
- Docker & Docker Compose  
- GitHub Actions (CI/CD)  
- Prometheus (Metrics Collection)  
- Grafana (Monitoring & Visualization)  

---

## Project Highlights

- Reinforcement Learning–based optimization system  
- Production-style AI deployment workflow  
- DevOps-integrated ML lifecycle  
- Real-time monitoring and observability  

---

## License
MIT License