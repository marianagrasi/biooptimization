# src/rbf.py
import numpy as np

SEED = 42


class RBFNetwork:
    def __init__(self, n_centers=10, sigma=1.0, lam=1e-3, seed=SEED):
        self.n_centers = n_centers
        self.sigma = sigma
        self.lam = lam
        self.seed = seed
        self.centers_, self.w_, self.b_ = None, None, None

    def _kmeans_centers(self, X):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=self.n_centers, replace=False)
        centers = X[idx].copy()
        for _ in range(25):
            d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=-1)
            assign = d.argmin(axis=1)
            for j in range(self.n_centers):
                pts = X[assign == j]
                if len(pts) > 0:
                    centers[j] = pts.mean(axis=0)
        return centers

    def _phi(self, X):
        d2 = np.sum((X[:, None, :] - self.centers_[None, :, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2 * self.sigma ** 2 + 1e-12))

    def fit(self, X, y):
        self.centers_ = self._kmeans_centers(X)
        Phi = self._phi(X)
        Phi_b = np.hstack([Phi, np.ones((len(X), 1))])
        I = np.eye(Phi_b.shape[1])
        I[-1, -1] = 0
        coef = np.linalg.solve(Phi_b.T @ Phi_b + self.lam * I, Phi_b.T @ y)
        self.w_, self.b_ = coef[:-1], coef[-1]
        return self

    def predict(self, X):
        Phi = self._phi(X)
        return Phi @ self.w_ + self.b_


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))