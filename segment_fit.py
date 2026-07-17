"""
Piecewise-linear COP fit for the inverter heat pump, parameterised by the number
of segments. Single source of truth for the segment slopes/intercepts used by the
PWL and PWLR investment models, so a segment-count sensitivity only changes one
number (``n_fit_segments``) and never a hand-transcribed table.

Fit method: **interpolating secants**. The smoothed datasheet performance map
``q(PLR) = COP(PLR) * PLR`` (normalised heat vs part-load ratio) is represented by
a shape-preserving PCHIP spline through the datasheet points; each segment is the
chord of that spline between its two equally spaced breakpoints. So every segment
touches the map exactly at its corners and the segments join continuously. At
``n=9`` the breakpoints coincide with the datasheet points, reproducing the
point-to-point 9-segment table used as the ex-post reference model.

Source of the datasheet curve: InvertedHP_data.ipynb (hand-smoothed COP).
Run ``python segment_fit.py`` to dump the 1..9-segment tables to
``cop_segment_fits.json`` for inspection.
"""
import json

import numpy as np
from scipy.interpolate import PchipInterpolator

# --- datasheet performance map: smoothed COP vs part-load ratio (from InvertedHP_data.ipynb) ---
_PLR_DATA = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
_COP_SMOOTH = np.array([2.515, 3.074, 3.284, 3.400, 3.456, 3.491, 3.518, 3.538, 3.552, 3.560])
_PLR_MIN = 0.1  # minimum part-load ratio spanned by the map / segments


def _q_curve():
    """Normalised heat output q = COP*PLR as a shape-preserving spline of PLR."""
    return PchipInterpolator(_PLR_DATA, _COP_SMOOTH * _PLR_DATA)


def _tightest_underestimator(bp, yb):
    """
    Highest single line L(PLR) = m*PLR + c that stays at or below the secant PWL.
    The PWL is linear between corners and L is linear, so L <= PWL everywhere iff
    L <= PWL at every corner. Returns (K_lu, D_lu) with q >= K_lu*PLR - D_lu.
    """
    x, y = np.asarray(bp, float), np.asarray(yb, float)
    best_slope, best_intercept = None, -np.inf
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if np.isclose(x[i], x[j]):
                continue
            m = (y[j] - y[i]) / (x[j] - x[i])
            c = float(np.min(y - m * x))  # max feasible intercept for this slope
            if np.all(m * x + c <= y + 1e-9) and c > best_intercept:
                best_slope, best_intercept = float(m), c
    return best_slope, -best_intercept  # D_lu is the positive offset


def fit_cop_segments(n_segments, plr_min=_PLR_MIN):
    """
    Secant piecewise-linear fit of the COP map with ``n_segments`` equally spaced
    segments over [plr_min, 1.0].

    Returns a dict:
      segments : ['s1', ..., 'sn']
      K, D_pos, D_neg, RMin, RMax : {seg: value}  with q = K*PLR + D_pos - D_neg on [RMin, RMax]
      K_lu, D_lu : tightest single linear underestimator of the fit (q >= K_lu*PLR - D_lu)
    """
    n = int(n_segments)
    if n < 1:
        raise ValueError(f"n_fit_segments must be >= 1, got {n_segments!r}")

    curve = _q_curve()
    bp = np.linspace(plr_min, 1.0, n + 1)
    yb = curve(bp)

    segs, K, D_pos, D_neg, RMin, RMax = [], {}, {}, {}, {}, {}
    for i in range(n):
        s = f"s{i + 1}"
        slope = (yb[i + 1] - yb[i]) / (bp[i + 1] - bp[i])
        intercept = yb[i] - slope * bp[i]
        segs.append(s)
        K[s] = float(slope)
        D_pos[s] = float(max(intercept, 0.0))
        D_neg[s] = float(max(-intercept, 0.0))
        RMin[s] = float(bp[i])
        RMax[s] = float(bp[i + 1])

    K_lu, D_lu = _tightest_underestimator(bp, yb)
    return {"segments": segs, "K": K, "D_pos": D_pos, "D_neg": D_neg,
            "RMin": RMin, "RMax": RMax, "K_lu": K_lu, "D_lu": D_lu}


def resolve_n_segments(*sources, default=4):
    """
    Read ``n_fit_segments`` from the first source that provides it (scenario params
    take precedence over global parameter.yaml). Sources may be dicts or pandas
    Series; missing / blank (NaN) entries are skipped. Falls back to ``default``.
    """
    for src in sources:
        if src is None or not hasattr(src, "get"):
            continue
        val = src.get("n_fit_segments")
        if val is None:
            continue
        try:
            if val != val:  # NaN
                continue
            return int(val)
        except (TypeError, ValueError):
            continue
    return default


if __name__ == "__main__":
    out = {}
    print(f"{'n':>2} | {'segment slopes K (s1..sn)':50} | {'underestimator (K_lu, D_lu)'}")
    for n in range(1, 10):
        fit = fit_cop_segments(n)
        out[str(n)] = fit
        ks = ", ".join(f"{fit['K'][s]:.3f}" for s in fit["segments"])
        print(f"{n:>2} | {ks:50} | ({fit['K_lu']:.4f}, {fit['D_lu']:.4f})")
    with open("cop_segment_fits.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote cop_segment_fits.json (n = 1..9).")
