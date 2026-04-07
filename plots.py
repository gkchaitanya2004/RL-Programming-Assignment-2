# All .npy files should be present in the same directory
# Each file should have shape: (15, 60)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde, nct, norm

def compute_ci(data):
    mean = np.mean(data, axis=0)
    ci = stats.sem(data, axis=0) * stats.t.ppf(0.975, df=data.shape[0]-1)
    return mean, mean - ci, mean + ci


# =========================================================
# Q2: Vanilla DQN Learning Curve
# Expected file:
# - returns_2000.npy
# =========================================================

data = np.load("returns_2000.npy")
mean, lo, hi = compute_ci(data)

plt.figure(figsize=(8,5))
plt.plot(mean, label="DQN")
plt.fill_between(range(len(mean)), lo, hi, alpha=0.2)

plt.title("Q2: DQN Learning Curve (Mean ± 95% CI)")
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# Q3: Effect of truncation length
# Expected files:
# - returns_200.npy
# - returns_1000.npy
# - returns_2000.npy
# =========================================================

files = {
    "200": "returns_200.npy",
    "1000": "returns_1000.npy",
    "2000": "returns_2000.npy"
}

for label in files:
    data = np.load(files[label])
    m, lo, hi = compute_ci(data)

    plt.figure(figsize=(8,5))
    plt.plot(m, label=f"T={label}")
    plt.fill_between(range(len(m)), lo, hi, alpha=0.15)

    plt.title(f"Q3:Truncation Length = {label}")
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# Q4(a): Replay Factor Comparison
# Expected files:
# - returns_rho1.npy
# - returns_rho2.npy
# - returns_rho4.npy
# - returns_rho8.npy
# =========================================================

RHO = [1, 2, 4, 8]

plt.figure(figsize=(9,5))

for rho in RHO:
    data = np.load(f"returns_rho{rho}.npy")
    m, lo, hi = compute_ci(data)

    plt.plot(m, label=f"ρ={rho}")
    plt.fill_between(range(len(m)), lo, hi, alpha=0.15)

plt.title("Q4(a): Replay Factor Comparison")
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --- AUC plot ---
means = []
cis = []

for rho in RHO:
    data = np.load(f"returns_rho{rho}.npy")
    per_run = np.mean(data, axis=1)

    m = np.mean(per_run)
    ci = 1.96 * np.std(per_run) / np.sqrt(len(per_run))

    means.append(m)
    cis.append(ci)

positions = np.arange(len(RHO))

plt.figure(figsize=(6,5))
plt.plot(positions, means, marker='o')
plt.errorbar(positions, means, yerr=cis, fmt='o', capsize=5)

plt.xticks(positions, [f"ρ={r}" for r in RHO])
plt.title("Q4(a): Aggregate Performance")
plt.ylabel("Mean Return")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# Q4(b): Distribution of Performance
# Expected files:
# - returns_rho1.npy
# - returns_rho2.npy
# - returns_rho4.npy
# - returns_rho8.npy
# =========================================================

plt.figure(figsize=(8,5))

for rho in RHO:
    data = np.load(f"returns_rho{rho}.npy")
    auc = np.mean(data, axis=1)

    kde = gaussian_kde(auc)
    x = np.linspace(auc.min()-200, auc.max()+200, 300)

    plt.plot(x, kde(x), label=f"ρ={rho}")
    plt.axvline(np.mean(auc), linestyle='--')
    plt.fill_between(x, kde(x), alpha = 0.15)

plt.title("Q4(b): Performance Distribution")
plt.xlabel("Aggregate Return")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# Q4(c): Tolerance Intervals
# Expected files:
# - returns_rho1.npy
# - returns_rho2.npy
# - returns_rho4.npy
# - returns_rho8.npy
# =========================================================

def tolerance_k(n, alpha=0.05, beta=0.9):
    z = norm.ppf(beta)
    return nct.ppf(1-alpha, df=n-1, nc=np.sqrt(n)*z) / np.sqrt(n)

k = tolerance_k(15)

plt.figure(figsize=(9,5))

for rho in RHO:
    data = np.load(f"returns_rho{rho}.npy")

    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)

    plt.plot(m, label=f"ρ={rho}")
    plt.fill_between(range(len(m)), m - k*s, m + k*s, alpha=0.1)

plt.title("Q4(c): Tolerance Intervals")
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# Q4(d): Sensitivity Analysis
# =========================================================

def compute_stats(file_dict):
    x_vals = sorted(file_dict.keys())
    means = []
    cis = []

    for x in x_vals:
        data = np.load(file_dict[x])
        per_run = np.mean(data, axis=1)

        mean = np.mean(per_run)
        std = np.std(per_run)

        ci = 1.96 * std / np.sqrt(len(per_run))

        means.append(mean)
        cis.append(ci)

    return x_vals, np.array(means), np.array(cis)


def plot_sensitivity(files_rho1, files_rho4, xlabel, title):
    x1, m1, c1 = compute_stats(files_rho1)
    x4, m4, c4 = compute_stats(files_rho4)

    positions = np.arange(len(x1))

    plt.figure(figsize=(8,5))

    plt.plot(positions, m1, marker='o', label="ρ = 1")
    plt.fill_between(positions, m1-c1, m1+c1, alpha=0.2)

    plt.plot(positions, m4, marker='o', label="ρ = 4")
    plt.fill_between(positions, m4-c4, m4+c4, alpha=0.2)

    plt.xticks(positions, x1)
    plt.ylim(-2000, -200)
    plt.xlabel(xlabel)
    plt.ylabel("Aggregate Performance")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------- Batch size sensitivity --------
# Expected files:
# - returns_rho1_bs16.npy, returns_rho1_bs24.npy, returns_2000.npy (bs=32)
# - returns_rho1_bs64.npy, returns_rho1_bs128.npy
# - returns_rho4_bs16.npy, returns_rho4_bs24.npy, returns_rho4.npy (bs=32)
# - returns_rho4_bs64.npy, returns_rho4_bs128.npy

files_rho1_bs = {
    16: "returns_rho1_bs16.npy",
    24: "returns_rho1_bs24.npy",
    32: "returns_2000.npy",
    64: "returns_rho1_bs64.npy",
    128: "returns_rho1_bs128.npy"
}

files_rho4_bs = {
    16: "returns_rho4_bs16.npy",
    24: "returns_rho4_bs24.npy",
    32: "returns_rho4.npy",
    64: "returns_rho4_bs64.npy",
    128: "returns_rho4_bs128.npy"
}

plot_sensitivity(files_rho1_bs, files_rho4_bs, "Batch Size", "Q4(d): Batch Size Sensitivity")


# -------- Target network sensitivity --------
# Expected files:
# - returns_rho1_tnr25.npy, returns_rho1_tnr40.npy, returns_2000.npy (tnr=50)
# - returns_rho1_tnr100.npy, returns_rho1_tnr200.npy
# - returns_rho4_tnr25.npy, returns_rho4_tnr40.npy, returns_rho4.npy (tnr=50)
# - returns_rho4_tnr100.npy, returns_rho4_tnr200.npy

files_rho1_tnr = {
    25: "returns_rho1_tnr25.npy",
    40: "returns_rho1_tnr40.npy",
    50: "returns_2000.npy",
    100: "returns_rho1_tnr100.npy",
    200: "returns_rho1_tnr200.npy"
}

files_rho4_tnr = {
    25: "returns_rho4_tnr25.npy",
    40: "returns_rho4_tnr40.npy",
    50: "returns_rho4.npy",
    100: "returns_rho4_tnr100.npy",
    200: "returns_rho4_tnr200.npy"
}

plot_sensitivity(files_rho1_tnr, files_rho4_tnr, "Target Update Frequency", "Q4(d): Target Network Sensitivity")