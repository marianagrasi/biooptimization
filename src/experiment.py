# src/experiment.py
import sys, os, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from aco import aco_tsp, nearest_neighbor_route, distance_matrix, route_length
from ssa import ssa_optimize, rastrigin
from rbf import RBFNetwork, mse
from db import get_db, save_run

SEED = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Cities.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_tsp_aco():
    print("\n" + "="*60)
    print("EJERCICIO 1 — ACO para TSP")
    print("="*60)
    df = pd.read_csv(DATA_PATH)
    tsp_df = df.head(15).copy()
    coords = tsp_df[["latitude", "longitude"]].values
    D = distance_matrix(coords)

    rng = np.random.default_rng(SEED)
    rand_len = np.mean([route_length(rng.permutation(len(D)), D) for _ in range(200)])
    nn_route = nearest_neighbor_route(D)
    nn_len = route_length(nn_route, D)

    t0 = time.time()
    aco_res = aco_tsp(D, seed=SEED)
    aco_time = time.time() - t0

    print(f"Ruta aleatoria (prom 200): {rand_len:.2f}")
    print(f"Vecino mas cercano:        {nn_len:.2f}")
    print(f"ACO:                       {aco_res['best_len']:.2f}  ({aco_time:.2f}s)")

    plt.figure()
    plt.plot(aco_res["best_curve"], label="Mejor global", linewidth=2)
    plt.plot(aco_res["mean_curve"], label="Media colonia", alpha=0.7)
    plt.xlabel("Iteracion"); plt.ylabel("Longitud")
    plt.title("Convergencia ACO - TSP (15 ciudades)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "aco_convergencia.png"), dpi=100)
    plt.close()


def run_ssa_rastrigin():
    print("\n" + "="*60)
    print("EJERCICIO 2 — SSA en Rastrigin")
    print("="*60)
    t0 = time.time()
    res = ssa_optimize(rastrigin, dim=10, bounds=(-5.12, 5.12),
                       n_iter=150, seed=SEED)
    t = time.time() - t0
    print(f"SSA Rastrigin (dim=10): {res['f_best']:.6f}  ({t:.2f}s)")

    plt.figure()
    plt.plot(res["best_curve"], linewidth=2)
    plt.yscale("log")
    plt.xlabel("Iteracion"); plt.ylabel("Mejor fitness (log)")
    plt.title("SSA en Rastrigin")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ssa_rastrigin.png"), dpi=100)
    plt.close()


def split_data(X, y, seed=SEED, train=0.6, val=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(train * len(X))
    n_val = int(val * len(X))
    tr, va, te = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


M_MAX = 40
GAMMA = 0.01
THETA_BOUNDS = (np.array([2, -2, -5]), np.array([M_MAX, 1, 0]))


def decode_theta(theta):
    M = int(np.clip(round(theta[0]), 2, M_MAX))
    return M, 10 ** theta[1], 10 ** theta[2]


def rbf_fitness(theta, Xtr, ytr, Xv, yv, seed=SEED):
    M, sigma, lam = decode_theta(theta)
    model = RBFNetwork(n_centers=M, sigma=sigma, lam=lam, seed=seed)
    model.fit(Xtr, ytr)
    return mse(yv, model.predict(Xv)) + GAMMA * (M / M_MAX)


def run_rbf_experiment(n_seeds=10):
    print("\n" + "="*60)
    print(f"EJERCICIO 3 — RBF vs RBF+SSA ({n_seeds} semillas)")
    print("="*60)
    df = pd.read_csv(DATA_PATH)
    X = df[["latitude", "longitude"]].values
    y = df["temperature"].values
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    y_scaled = (y - y.mean()) / y.std()

    records = []
    for s in range(n_seeds):
        (Xtr, ytr), (Xv, yv), (Xt, yt) = split_data(X_scaled, y_scaled, seed=s)

        t0 = time.time()
        base = RBFNetwork(n_centers=10, sigma=1.0, lam=1e-3, seed=s).fit(Xtr, ytr)
        bt = time.time() - t0
        base_mse = mse(yt, base.predict(Xt))

        records.append({
            "algorithm": "rbf_baseline", "seed": s,
            "dataset": "cities_temperature",
            "params": {"M": 10, "sigma": 1.0, "lambda": 1e-3},
            "metrics": {"mse": float(base_mse), "time_s": float(bt)},
            "evaluations": 0, "convergence": [],
            "timestamp": datetime.now(timezone.utc),
        })

        t0 = time.time()
        f = lambda th: rbf_fitness(th, Xtr, ytr, Xv, yv, seed=s)
        r = ssa_optimize(f, dim=3, bounds=THETA_BOUNDS,
                         n_pop=15, n_iter=40, seed=s)
        st = time.time() - t0
        M, sig, lam = decode_theta(r["x_best"])
        tuned = RBFNetwork(n_centers=M, sigma=sig, lam=lam, seed=s).fit(Xtr, ytr)
        tuned_mse = mse(yt, tuned.predict(Xt))

        records.append({
            "algorithm": "ssa_rbf", "seed": s,
            "dataset": "cities_temperature",
            "params": {"M": int(M), "sigma": float(sig), "lambda": float(lam)},
            "metrics": {"mse": float(tuned_mse), "time_s": float(st)},
            "evaluations": 15*40,
            "convergence": r["best_curve"].tolist(),
            "timestamp": datetime.now(timezone.utc),
        })
        print(f"  seed={s:2d} base={base_mse:.4f} ssa={tuned_mse:.4f} "
              f"M={M} sigma={sig:.3f} lambda={lam:.5f}")
    return records


def main():
    print("\nIniciando experimentos...")
    run_tsp_aco()
    run_aco_configs()
    run_ssa_rastrigin()
    records = run_rbf_experiment(n_seeds=10)

    print("\n" + "="*60)
    print("GUARDANDO EN MONGODB ATLAS")
    print("="*60)
    try:
        col = get_db()
        col.delete_many({})
        for rec in records:
            save_run(col, rec)
        print(f"[OK] {len(records)} documentos guardados en Atlas")
    except Exception as e:
        print(f"[AVISO] Error Mongo: {e}")

    df = pd.DataFrame([{
        "algorithm": r["algorithm"], "seed": r["seed"],
        "mse": r["metrics"]["mse"], "time_s": r["metrics"]["time_s"],
        "M": r["params"]["M"], "sigma": r["params"]["sigma"],
        "lambda": r["params"]["lambda"],
    } for r in records])
    df.to_csv(os.path.join(RESULTS_DIR, "resultados_rbf.csv"), index=False)
    print("CSV en results/resultados_rbf.csv")
    print("\nRESUMEN:")
    print(df.groupby("algorithm")["mse"].agg(["mean","std","median","min","max"]))

    base_s = df[df.algorithm == "rbf_baseline"]["mse"].values
    tuned_s = df[df.algorithm == "ssa_rbf"]["mse"].values
    plt.figure()
    plt.boxplot([base_s, tuned_s], tick_labels=["Baseline", "RBF+SSA"])
    plt.ylabel("MSE en test"); plt.title("Distribucion MSE — 10 semillas")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "boxplot.png"), dpi=100)
    plt.close()
    print("[OK] Terminado. Revisa results/")

    #Segunda configuracion de ACO
def run_aco_configs():
    """Prueba diferentes configuraciones de ACO y compara convergencia."""
    print("\n" + "="*60)
    print("EJERCICIO 1b — Comparación de configuraciones ACO")
    print("="*60)
    df = pd.read_csv(DATA_PATH)
    tsp_df = df.head(15).copy()
    coords = tsp_df[["latitude", "longitude"]].values
    D = distance_matrix(coords)

    configs = {
        "Balanceado (a=1, b=3, rho=0.5)":      dict(alpha=1.0, beta=3.0, rho=0.5),
        "Mas explotacion (a=3, b=1, rho=0.7)": dict(alpha=3.0, beta=1.0, rho=0.7),
        "Mas exploracion (a=0.5, b=1, rho=0.2)": dict(alpha=0.5, beta=1.0, rho=0.2),
    }

    curves = {}
    for label, params in configs.items():
        r = aco_tsp(D, n_iter=150, seed=SEED, **params)
        curves[label] = r["best_curve"]
        print(f"  {label:42s} -> {r['best_len']:.2f}")

    plt.figure()
    for label, curve in curves.items():
        plt.plot(curve, label=label, linewidth=2)
    plt.xlabel("Iteracion")
    plt.ylabel("Mejor longitud")
    plt.title("Efecto de alpha, beta y rho en ACO")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "aco_configs.png"), dpi=100)
    plt.close()
    print("  Guardado: results/aco_configs.png")

if __name__ == "__main__":
    main()