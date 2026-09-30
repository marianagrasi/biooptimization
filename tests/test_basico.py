import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from aco import aco_tsp, distance_matrix
from ssa import ssa_optimize, rastrigin
from rbf import RBFNetwork


def test_aco_valid_permutation():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(10, 2))
    D = distance_matrix(coords)
    r = aco_tsp(D, n_iter=20)["best_route"]
    assert sorted(r.tolist()) == list(range(10))


def test_ssa_bounds():
    lb, ub = np.array([-5.12]*5), np.array([5.12]*5)
    r = ssa_optimize(rastrigin, dim=5, bounds=(lb, ub), n_iter=30)
    assert np.all(r["x_best"] >= lb)
    assert np.all(r["x_best"] <= ub)


def test_rbf_no_nan():
    X = np.random.randn(50, 2)
    y = np.random.randn(50)
    m = RBFNetwork(n_centers=6, sigma=1.0, lam=1e-2).fit(X, y)
    Phi = m._phi(X)
    assert Phi.shape == (50, 6)
    assert np.isfinite(Phi).all()


def test_reproducibility():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(10, 2))
    D = distance_matrix(coords)
    a = aco_tsp(D, n_iter=30, seed=123)["best_len"]
    b = aco_tsp(D, n_iter=30, seed=123)["best_len"]
    assert np.isclose(a, b)