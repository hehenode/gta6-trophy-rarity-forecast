"""gta6-trophy-rarity-forecast-01 entry: a local growth-curve forecaster.

For every trophy, the earn rate (the percent of Exophase's tracked players who
hold it) is forecast at the case's target as five quantiles, scored by pinball
loss on the logit scale.

How it works
------------
The page the house records is a step function: Exophase refreshes it every few
hours, so consecutive identical reads are collapsed into one page state that
lasts until the next state begins. States from before the launch transient are
not the launch population (a few hundred pre-launch syncs), so the fit uses
states that begin after it.

A trophy's rate moves slowly while the tracked crowd is young, then climbs
faster as that crowd ages into the trophy's typical completion time, so the
extrapolation that holds up is a local one: a weighted least-squares line of
the logit rate against x = ln(1 + hours since release) over the last day of
page states. The weights carry three things: how long each state was the page,
a recency decay, and counting precision -- a rate printed from few players is
a noisy logit, with variance about 1 / (players * p * (1 - p)). Each trophy's
slope is shrunk toward the cross-trophy pool (empirical Bayes), which steadies
the rare trophies whose own slope is mostly rounding steps. The median is the
shrunk line read at the target, kept within a rail of the last rate.

The spread is honest about what the fit cannot see: the fit's residual
dispersion, its slope uncertainty magnified by the extrapolation distance, and
a horizon term that grows like a random walk's. It is applied on the logit
scale, the scale the scoring uses.

When a case has no post-launch history to fit (the practice cases are cut from
the frozen pre-launch page), the forecast is the last rate with a wide spread
and, while the tracked base is still a small pre-launch cohort, a dilution
haircut: a launch floods the base with new players who hold almost nothing, so
rates fall before they climb. A trophy every tracked player has always had is
structural (earned in the opening minutes) and is exempt from the haircut.

Deterministic: pure arithmetic on the case file, no clocks, no randomness.
Runs as the grader runs it: python3 -P -s /bot/main.py, standard library only,
reading $MAKEFUN_INPUT_DIR (else /input) and writing $MAKEFUN_OUTPUT_DIR
(else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import (this entry is one file, but the line
# is harmless and keeps the layout standard).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

WINDOW_H = 24.0            # fit window: page states starting within a day of the checkpoint
HALF_LIFE_H = 20.0         # recency half-life inside the window
SHRINK_FLOOR = 0.08        # min prior variance for the slope pool
SPREAD_FLOOR = 0.82        # min logit spread for any trophy
RW_PER_SQRT_DAY = 0.62     # horizon-driven spread component
RAIL = 4.5                 # max logit move of the median away from the last rate


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up; a whole number comes back as an int."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000
    return int(v) if float(v).is_integer() else v


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def release_epoch(case):
    rd = (case.get("game") or {}).get("release_date") or ""
    try:
        return datetime.fromisoformat(rd).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return parse_ts(case["history"][0]["at"])


def x_of(ts, rel):
    return math.log(1.0 + max(0.0, (ts - rel) / 3600.0))


def collapse_states(history):
    """Consecutive reads showing the same page are one state; a state lasts
    until the next state begins. [(start_ts, players, rates, duration_h)]"""
    states = []
    for rd in history:
        key = (rd["players_tracked"], tuple(rd["rates"]))
        ts = parse_ts(rd["at"])
        if states and states[-1][3] == key:
            continue
        states.append([ts, rd["players_tracked"], rd["rates"], key])
    out = []
    for i, st in enumerate(states):
        dur = (states[i + 1][0] - st[0]) / 3600.0 if i + 1 < len(states) else 0.0
        out.append((st[0], st[1], st[2], max(dur, 0.05)))
    return out


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b x.
    Returns (a, b, xbar, ybar, sum_w, sxx, resid_sd) or None."""
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (ybar, 0.0, xbar, ybar, sw, sxx, 0.0)
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw)
    return (a, b, xbar, ybar, sw, sxx, resid)


def fit_over(states, i, rel, at):
    """The local logit fit for trophy i over the given page states."""
    xs, ys, ws = [], [], []
    for ts, players, rates, dur in states:
        r = rates[i]
        if r is None:
            continue
        p = clamp_pct(r) / 100.0
        prec = min(players * p * (1.0 - p), 400.0)
        rec = 0.5 ** (max(0.0, (at - ts) / 3600.0) / HALF_LIFE_H)
        xs.append(x_of(ts, rel))
        ys.append(logit(r))
        ws.append((0.05 + prec) * min(dur, 6.0) * rec)
    return wls_line(xs, ys, ws) if len(xs) >= 3 else None


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    history = case["history"]
    if not history:
        raise ValueError("no history")
    rel = release_epoch(case)
    at = parse_ts(case["at"])
    target = parse_ts(case["target"])
    x_tgt = x_of(target, rel)
    days = case["horizon_minutes"] / 1440.0
    states = collapse_states(history)
    last = history[-1]
    n_last = last["players_tracked"]

    # Post-regime states: page states that begin after the launch transient.
    post = [s for s in states if s[0] >= rel + 3600.0]
    regime_ok = len(post) >= 3
    base_states = post if regime_ok else states
    window = [s for s in base_states if s[0] >= at - WINDOW_H * 3600.0]
    if len(window) < 3:
        window = base_states

    fits = [fit_over(window, i, rel, at) for i in range(len(case["trophies"]))]
    slopes = [f[1] for f in fits if f is not None]
    if slopes:
        pool_mean = sum(slopes) / len(slopes)
        pool_var = sum((s - pool_mean) ** 2 for s in slopes) / len(slopes)
    else:
        pool_mean, pool_var = 0.0, 1.0
    pool_var = max(pool_var, SHRINK_FLOOR)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r_last = last["rates"][i]
        r_last = r_last if r_last is not None else 0.0
        fit = fits[i]
        if fit is None or not regime_ok:
            # Flat / pre-launch regime: honest wide spread, mild dilution
            # haircut while the tracked base is a small pre-launch cohort.
            lg = logit(r_last)
            shift = -0.9 if (n_last <= 5000 and at - rel <= 24 * 3600) else 0.0
            s = 1.35 + 0.30 * math.sqrt(days)
            hist_rates = [rd["rates"][i] for rd in history if rd["rates"][i] is not None]
            if r_last >= 99.99 and hist_rates and min(hist_rates) >= 99.9:
                # A trophy every tracked player has always had is structural
                # (earned in the opening minutes), not a dilution candidate.
                # Keep the regime's honest wide spread rather than pinning a
                # narrow one: certainty here is about the level, not the band.
                lg, shift = logit(99.99), 0.0
                s = max(s, 0.8)
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(lg + shift + z * s)) for z in Z]})
            continue
        a, b, xbar, ybar, sw, sxx, resid = fit
        if sxx > 1e-9:
            var_b = max((resid * resid) / sxx if resid > 0 else 0.25 / sxx, 1e-4)
        else:
            var_b = 4.0
        b_shrunk = (b / var_b + pool_mean / pool_var) / (1.0 / var_b + 1.0 / pool_var)
        a_shrunk = ybar - b_shrunk * xbar
        lg_med = a_shrunk + b_shrunk * x_tgt
        lg_last = logit(r_last)
        lg_med = min(max(lg_med, lg_last - RAIL), lg_last + RAIL)
        s_param = math.sqrt(var_b) * abs(x_tgt - xbar)
        s = math.sqrt(resid * resid + s_param * s_param
                      + (RW_PER_SQRT_DAY * math.sqrt(days)) ** 2)
        s = max(s, SPREAD_FLOOR)
        recent = [rates[i] for _, _, rates, _ in window if rates[i] is not None]
        if r_last >= 99.99 and recent and min(recent) >= 99.9:
            lg_med = logit(99.99)
            s = max(s, 0.8)
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(lg_med + z * s)) for z in Z]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
