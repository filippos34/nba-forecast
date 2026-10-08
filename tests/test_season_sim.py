"""season_sim: per-game arithmetic identical to the daily forecasts; probabilities sum correctly."""
import itertools

import numpy as np
import pandas as pd

import predict
import season_sim as ss
from conftest import needs_data


def test_game_prob_matches_predict_base_prob():
    """Including the cases where availability flips who is 'favoured' (Elo decides home court)."""
    for h, a, ha, aa, neu, hr, ar in itertools.product([1494.0, 1576.0, 1632.5], [1500.0, 1613.7],
                                                       [339.4, 473.5], [392.9, 517.9], [False, True],
                                                       [0, 1, 3], [0, 2]):
        want, _ = predict._base_prob(h, a, hr, ar, neu, ha, aa)
        rest = predict._rest_elo(hr, ar, ss.br.PRODUCTION)
        got = float(ss.game_prob(np.array([h]), np.array([a]), np.array([ha]), np.array([aa]),
                                 np.array([neu]), np.array([rest]))[0])
        assert abs(got - want) < 1e-12, (h, a, ha, aa, neu, hr, ar)


@needs_data
def test_simulation_known_structure():
    teams = sorted(ss.EAST) + ["DAL", "DEN", "GS", "HOU", "LAC", "LAL", "MEM", "MIN", "NO", "OKC", "PHX",
                                "POR", "SA", "SAC", "UTAH"]
    inputs = pd.DataFrame({"elo": 1500.0, "avail": 400.0}, index=teams)
    inputs.loc["DEN", "elo"] = 1800.0                      # one dominant team
    t = ss.simulate(n=2000, seed=1, inputs=inputs)
    assert abs(t["p_champion"].sum() - 1) < 1e-9 and abs(t["p_east_or_west"].sum() - 2) < 1e-9
    assert t.iloc[0]["team"] == "DEN" and t.iloc[0]["p_champion"] > 0.3
    assert abs(t["mean_wins"].sum() - len(pd.read_csv(ss.DATA / "schedule_2026_27.csv"))) < 1
