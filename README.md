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

## Virtual Environment

Create a new conda environment. Open the anaconda prompt and write the following command
```
conda create --name FLenv python=3.11.12 numpy scipy jupyter
```
Once it has finished installing the packages activate the environment with
```
conda activate FLenv
```
Next, go to the [pytorch website](https://pytorch.org/get-started/locally/), select the appropriate system specifications and select the pip package manager. Copy the given command and go back to the anaconda prompt and run the command. An example is given below:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```
Now we are ready to install the flower framework: we need the simulation package as well as the vision dataset:
```
pip install "flwr[simulation]" "flwr-datasets[vision]"
```

## Experiments

### 1. Own Implementation of FedAvg vs Centralised

### 2. Federated Learning Using [Flower](https://flower.dev) framework

### 3. Dealing with IID and Non-IID
  - IID vs Non-IID using FedAvg vs SCAFFOLD vs FedPer vs Data Sharing 
  - IID vs Non-IID using FedPer 
  - IID vs Non-IID using SCAFFOLD 
  - IID vs Non-IID using Data Sharing 

### 4. Privacy and Security in FL
  - Differential privacy defense experiments
  - Gradient Inversion

### 5. Project?

---

## Report

The full LaTeX report with figures and references is located in the `report/` directory.

---

## 📚 References

This work is based on and extends from:
- McMahan et al., 2017: *Communication-Efficient Learning of Deep Networks from Decentralized Data*
- Beutel et al., 2022: *Flower: A Friendly Federated Learning Framework*
- Kairouz et al., 2021: *Advances and Open Problems in Federated Learning*

See `report/references.bib` for the full list of citations.

---

