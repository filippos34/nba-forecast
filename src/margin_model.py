"""
src/margin_model.py — point margin ~ Normal(μ, σ)  (Phase 2.1)

DERIVED (production, Gate 2): μ = σ · Φ⁻¹(p) from the model's win probability p, so the
moneyline, spread and alternate lines always agree. Only σ is fit (maximum likelihood on
margins of the training seasons). See DerivedSpread.

STANDALONE (comparison only): Elo difference → μ by a fitted slope.
===============================================================================
μ = slope × (home − away Elo, incl. home court / rest / availability) / 28,
margin ~ Normal(μ, σ); P(home win) = Φ(μ/σ); fair spread = −μ; P(home covers line L)
= Φ((μ + L)/σ). slope and σ are fit by least squares on the training seasons only.
The Elo walk itself (ratings, updates) is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

POINTS_TO_ELO = 28.0


@dataclass(frozen=True)
class MarginModel:
    slope: float
    sigma: float

    def mu(self, diff):
        return self.slope * np.asarray(diff, float) / POINTS_TO_ELO

    def p_win(self, diff):
        return norm.cdf(self.mu(diff) / self.sigma)

    def fair_spread(self, diff):
        """Home spread in the usual convention (negative = home favoured)."""
        return -self.mu(diff)

    def p_cover(self, diff, home_line):
        """P(home margin + home_line > 0), e.g. home −5.5 → home_line = −5.5."""
        return norm.cdf((self.mu(diff) + home_line) / self.sigma)


def fit(diff, margin) -> MarginModel:
    x = np.asarray(diff, float) / POINTS_TO_ELO
    y = np.asarray(margin, float)
    slope = float(x @ y / (x @ x))            # through the origin: diff already includes home court
    sigma = float(np.sqrt(np.mean((y - slope * x) ** 2)))
    return MarginModel(slope=slope, sigma=sigma)


@dataclass(frozen=True)
class DerivedSpread:
    """margin ~ Normal(σ·Φ⁻¹(p), σ). P(home win) is exactly p."""
    sigma: float

    def mu(self, p):
        return self.sigma * norm.ppf(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))

    def fair_spread(self, p):
        return -self.mu(p)

    def p_cover(self, p, home_line):
        return norm.cdf((self.mu(p) + home_line) / self.sigma)

    def p_over(self, p, total_mu, total_sigma, line):
        raise NotImplementedError("totals need a points model; not part of the win-prob model")


def fit_sigma(p, margin) -> DerivedSpread:
    """Maximum-likelihood σ for margin ~ N(σ z, σ²), z = Φ⁻¹(p)."""
    from scipy.optimize import minimize_scalar
    z = norm.ppf(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    y = np.asarray(margin, float)
    nll = lambda s: len(y) * np.log(s) + np.sum((y - s * z) ** 2) / (2 * s * s)
    return DerivedSpread(sigma=float(minimize_scalar(nll, bounds=(5, 30), method="bounded").x))


def crps_normal(y, mu, sigma):
    """CRPS of N(mu, sigma) at y (lower is better)."""
    z = (np.asarray(y, float) - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
