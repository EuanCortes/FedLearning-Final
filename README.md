# Federated Learning: Theory, Implementation, and Application

This repository accompanies a technical report exploring Federated Learning (FL) with a focus on the **Federated Averaging (FedAvg)** algorithm, alongside other algorithms such as **FedPer** and **SCAFFOLD**. We present both a from-scratch implementation and an application using the **Flower framework**, with comparisons against centralized training. The study includes experiments under IID and non-IID conditions on CIFAR-10 and methods for dealing with these scenarios.

Additionally, we explore privacy-preserving techniques and threat models in the federated setting. This includes implementations of **membership inference attacks**, **gradient leakage (DLG/iDLG)**, and defense mechanisms such as **differential privacy**.

---

## Repository Structure

...
│  
├── code/                - FedAvg (scratch & Flower), FedPer, SCAFFOLD, and utilities 
├── notebooks/           - Jupyter notebooks for exploratory analysis  
├── experiments/         - Scripts for running experiments  
├── results/             - Output plots, tables, logs  
├── report/              - LaTeX source of the written report  
├── data/                - Dataset Loaders partitioning and preprocessing  


## Experiments

### 1. Centralized vs Federated Training on CIFAR-10
- Compare FedAvg with a global model trained on the full dataset
- Evaluate convergence and accuracy under IID partitioning
- Modular and scalable simulation using the [Flower](https://flower.dev) framework

### 2. FedAvg under Non-IID Conditions
- Simulated non-IID splits (1-class and 2-class per client)
- Comparison of FedAvg, FedPer, and SCAFFOLD
- Evaluation of personalization and convergence behavior

### 3. Privacy and Security in FL
- Implementation of membership inference attacks
- Deep Leakage from Gradients (DLG/iDLG)
- Differential privacy defense experiments

### 4. THINGS-EEG Project
- Application of federated learning to decentralized EEG data

---

## Report

The full LaTeX report with figures and references is located in the `report/` directory.

Topics include:
- FedAvg implementation and theory
- Experimental results (IID and non-IID)
- Differential privacy and attack resilience
- Application to EEG data

---

## 📚 References

This work is based on and extends from:
- McMahan et al., 2017: *Communication-Efficient Learning of Deep Networks from Decentralized Data*
- Beutel et al., 2022: *Flower: A Friendly Federated Learning Framework*
- Kairouz et al., 2021: *Advances and Open Problems in Federated Learning*

See `report/references.bib` for the full list of citations.

---

