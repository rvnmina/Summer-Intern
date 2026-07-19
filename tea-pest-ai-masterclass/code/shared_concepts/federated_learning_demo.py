"""
federated_learning_demo.py
==========================
Federated learning (Paper 2) from scratch: train ONE model across several
simulated "tea estates" WITHOUT ever centralising their raw data. Only model
weights are averaged (FedAvg). Also shows FedProx, which stabilises training
when the estates' data is non-IID (statistically different from each other).

Run:  python federated_learning_demo.py

What you should see
-------------------
- Non-IID clients (each estate biased toward different classes).
- FedAvg global accuracy improving each round from averaged local models.
- FedProx doing at least as well under the non-IID skew.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLIENTS = 4          # 4 tea estates
N_CLASSES = 5          # 5 pests
N_FEATURES = 20
ROUNDS = 12
LOCAL_EPOCHS = 3
DEVICE = "cpu"
torch.manual_seed(0)


# ----------------------------------------------------------------------------
# A tiny shared model architecture.
# ----------------------------------------------------------------------------
def make_model():
    return nn.Sequential(nn.Linear(N_FEATURES, 32), nn.ReLU(), nn.Linear(32, N_CLASSES))


# ----------------------------------------------------------------------------
# Build NON-IID client datasets: each estate mostly sees a couple of classes,
# reproducing Paper 2's "non-IID plantation" challenge.
# ----------------------------------------------------------------------------
def make_non_iid_clients(n_per_client=200):
    # Fixed "true" decision boundary shared across estates.
    W_true = torch.randn(N_FEATURES, N_CLASSES)
    clients = []
    for c in range(N_CLIENTS):
        # Each estate over-samples 2 favoured classes (skewed distribution).
        favoured = [(c) % N_CLASSES, (c + 1) % N_CLASSES]
        X = torch.randn(n_per_client, N_FEATURES)
        logits = X @ W_true
        # Bias the labels toward this estate's favoured classes.
        logits[:, favoured] += 2.0
        y = logits.argmax(1)
        clients.append((X, y))
    # A shared held-out test set (balanced) to measure the GLOBAL model.
    Xte = torch.randn(400, N_FEATURES)
    yte = (Xte @ W_true).argmax(1)
    return clients, (Xte, yte)


# ----------------------------------------------------------------------------
# Local training at one client. Optionally add the FedProx proximal penalty
# (mu/2)*||theta - theta_global||^2 to keep the local model near the global one.
# ----------------------------------------------------------------------------
def local_train(global_state, data, epochs=LOCAL_EPOCHS, lr=0.05, mu=0.0):
    model = make_model()
    model.load_state_dict(copy.deepcopy(global_state))
    global_params = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    X, y = data
    for _ in range(epochs):
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        if mu > 0:  # FedProx penalty
            prox = sum(((p - gp) ** 2).sum() for p, gp in zip(model.parameters(), global_params))
            loss = loss + (mu / 2) * prox
        opt.zero_grad(); loss.backward(); opt.step()
    return model.state_dict(), len(y)


# ----------------------------------------------------------------------------
# FedAvg: data-weighted average of client weights  (Paper 2, §3.3)
#   theta_global = sum_k (n_k / n) * theta_k
# ----------------------------------------------------------------------------
def fedavg(client_states, client_sizes):
    total = sum(client_sizes)
    avg = copy.deepcopy(client_states[0])
    for key in avg:
        avg[key] = sum(cs[key] * (n / total) for cs, n in zip(client_states, client_sizes))
    return avg


@torch.no_grad()
def evaluate(state, test):
    model = make_model(); model.load_state_dict(state)
    X, y = test
    return (model(X).argmax(1) == y).float().mean().item()


# ----------------------------------------------------------------------------
# The federated training loop.
# ----------------------------------------------------------------------------
def run_federated(mu=0.0, label="FedAvg"):
    clients, test = make_non_iid_clients()
    global_state = make_model().state_dict()   # server's initial model
    print(f"\n--- {label} (mu={mu}) over {N_CLIENTS} non-IID estates ---")
    for r in range(1, ROUNDS + 1):
        client_states, client_sizes = [], []
        for data in clients:                    # each estate trains LOCALLY
            state, n = local_train(global_state, data, mu=mu)
            client_states.append(state); client_sizes.append(n)
        global_state = fedavg(client_states, client_sizes)   # server averages
        if r % 3 == 0 or r == 1:
            print(f"  round {r:2d}: global test accuracy = {evaluate(global_state, test):.3f}")
    return evaluate(global_state, test)


def main():
    print("=" * 70)
    print("FEDERATED LEARNING: raw estate data never leaves the estate.")
    print("Only model weights are shared and averaged by the server.")
    print("=" * 70)
    acc_avg = run_federated(mu=0.0, label="FedAvg")
    acc_prox = run_federated(mu=0.05, label="FedProx (drift-corrected)")
    print("\nFinal global-model accuracy:")
    print(f"  FedAvg : {acc_avg:.3f}")
    print(f"  FedProx: {acc_prox:.3f}  (proximal term helps under non-IID skew)")
    print("\nKey point: one shared model learned from 4 estates, zero raw-data sharing.")


if __name__ == "__main__":
    main()
