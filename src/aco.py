# src/aco.py
import numpy as np


def route_length(route, D):
    route = np.asarray(route)
    return D[route, np.roll(route, -1)].sum()


def distance_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def nearest_neighbor_route(D, start=0):
    n = D.shape[0]
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[current, j])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return np.array(route)


def aco_tsp(D, n_ants=20, n_iter=150, alpha=1.0, beta=3.0,
            rho=0.5, Q=100.0, tau0=1.0, seed=42):
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    tau = np.full((n, n), tau0)
    with np.errstate(divide="ignore"):
        eta = np.where(D > 0, 1.0 / D, 0.0)

    best_route, best_len = None, np.inf
    best_curve = np.zeros(n_iter)
    mean_curve = np.zeros(n_iter)

    for it in range(n_iter):
        routes = np.zeros((n_ants, n), dtype=int)
        lengths = np.zeros(n_ants)

        for k in range(n_ants):
            unvisited = list(range(n))
            start = rng.integers(0, n)
            unvisited.remove(start)
            route = [start]
            current = start
            while unvisited:
                weights = (tau[current, unvisited] ** alpha) * \
                          (eta[current, unvisited] ** beta)
                total = weights.sum()
                probs = weights / total if total > 0 \
                    else np.ones(len(unvisited)) / len(unvisited)
                choice = rng.choice(len(unvisited), p=probs)
                nxt = unvisited.pop(choice)
                route.append(nxt)
                current = nxt
            routes[k] = route
            lengths[k] = route_length(np.array(route), D)

        it_best = lengths.argmin()
        if lengths[it_best] < best_len:
            best_len = lengths[it_best]
            best_route = routes[it_best].copy()

        tau *= (1 - rho)
        for k in range(n_ants):
            r = routes[k]
            deposit = Q / lengths[k]
            tau[r, np.roll(r, -1)] += deposit
            tau[np.roll(r, -1), r] += deposit

        best_curve[it] = best_len
        mean_curve[it] = lengths.mean()

    return {
        "best_route": best_route,
        "best_len": best_len,
        "best_curve": best_curve,
        "mean_curve": mean_curve,
    }