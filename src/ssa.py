# src/ssa.py
import numpy as np


def ssa_optimize(f, dim, bounds, n_pop=30, n_iter=120, pd_ratio=0.2,
                 sd_ratio=0.2, ST=0.8, seed=42):
    rng = np.random.default_rng(seed)
    lb = np.broadcast_to(bounds[0], dim).astype(float)
    ub = np.broadcast_to(bounds[1], dim).astype(float)

    X = rng.uniform(lb, ub, size=(n_pop, dim))
    fitness = np.array([f(x) for x in X])

    n_producers = max(1, int(pd_ratio * n_pop))
    n_scouts = max(1, int(sd_ratio * n_pop))

    best_curve = np.zeros(n_iter)
    mean_curve = np.zeros(n_iter)
    best_idx = fitness.argmin()
    X_best, f_best = X[best_idx].copy(), fitness[best_idx]

    for t in range(n_iter):
        order = np.argsort(fitness)
        X = X[order]
        fitness = fitness[order]
        X_best_t, X_worst_t = X[0].copy(), X[-1].copy()

        alpha_ = rng.uniform(0.5, 1.0)
        R2 = rng.uniform(0, 1)
        X_new = X.copy()

        for i in range(n_producers):
            if R2 < ST:
                X_new[i] = X[i] * np.exp(-(i + 1) / (alpha_ * n_iter + 1e-9))
            else:
                Q = rng.normal(0, 1)
                X_new[i] = X[i] + Q * np.ones(dim)

        for i in range(n_producers, n_pop):
            if i > n_pop / 2:
                Q = rng.normal(0, 1)
                X_new[i] = Q * np.exp((X_worst_t - X[i]) / ((i + 1) ** 2))
            else:
                A = rng.choice([-1, 1], size=dim)
                X_new[i] = X_new[n_producers - 1] + \
                    np.abs(X[i] - X_new[n_producers - 1]) * A

        scout_idx = rng.choice(n_pop, size=n_scouts, replace=False)
        for i in scout_idx:
            beta_ = rng.normal(0, 1)
            if fitness[i] > f_best:
                X_new[i] = X_best_t + beta_ * np.abs(X[i] - X_best_t)
            else:
                K = rng.uniform(-1, 1)
                denom = np.abs(fitness[i] - fitness[-1]) + 1e-12
                X_new[i] = X[i] + K * (X[i] - X_worst_t) / denom

        X = np.clip(X_new, lb, ub)
        fitness = np.array([f(x) for x in X])

        cur_best_idx = fitness.argmin()
        if fitness[cur_best_idx] < f_best:
            f_best = fitness[cur_best_idx]
            X_best = X[cur_best_idx].copy()

        best_curve[t] = f_best
        mean_curve[t] = fitness.mean()

    return {
        "x_best": X_best,
        "f_best": f_best,
        "best_curve": best_curve,
        "mean_curve": mean_curve,
    }


def rastrigin(x):
    d = len(x)
    return float(10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))