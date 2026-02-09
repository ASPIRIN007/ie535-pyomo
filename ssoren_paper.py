import numpy as np
from scipy.optimize import linprog

def solve_PS(players, T, params, fix_initial_energy=True):
    """
    Solve the paper's (PS) LP for a coalition S=players over periods T.

    params expects:
      cG, cInv, cEn : floats
      p[t], h[t]    : list length |T|
      d[i][t]       : dict player -> list length |T|
      ghat[i][t]    : dict player -> list length |T|
      gtilde[i][t]  : dict player -> list length |T|
      s_chg[i], s_dis[i] : dict player -> float (existing limits per period)
      Emin[i], Ebar[i]   : dict player -> float
      eta_c[i], eta_d[i] : dict player -> float
    """
    I = len(players)
    TT = len(T)

    # ---------- variable indexing ----------
    # Order:
    # xG[i], xInv[i], xEn[i],
    # g[i,t], sc[i,t], sd[i,t], u[i,t] (u = energy ABOVE Emin, so nonnegative),
    # z[t], w[t]
    idx = {}
    n = 0

    def add(name, shape):
        nonlocal n
        size = int(np.prod(shape))
        idx[name] = (n, shape)
        n += size

    add("xG",   (I,))
    add("xInv", (I,))
    add("xEn",  (I,))
    add("g",    (I, TT))
    add("sc",   (I, TT))
    add("sd",   (I, TT))
    add("u",    (I, TT))   # u = e - Emin  => u >= 0
    add("z",    (TT,))
    add("w",    (TT,))

    def flat(name, i=None, t=None):
        start, shape = idx[name]
        if len(shape) == 1:
            if shape[0] == I:
                return start + i
            elif shape[0] == TT:
                return start + t
            else:
                raise ValueError("Unexpected 1D shape")
        return start + i*TT + t

    # ---------- objective ----------
    c = np.zeros(n)
    c[idx["xG"][0]:idx["xG"][0]+I]     = params["cG"]
    c[idx["xInv"][0]:idx["xInv"][0]+I] = params["cInv"]
    c[idx["xEn"][0]:idx["xEn"][0]+I]   = params["cEn"]

    c[idx["z"][0]:idx["z"][0]+TT] = np.array(params["p"], dtype=float)
    c[idx["w"][0]:idx["w"][0]+TT] = -np.array(params["h"], dtype=float)

    bounds = [(0, None)] * n  # all vars >= 0

    A_eq, b_eq = [], []
    A_ub, b_ub = [], []

    # ---------- energy balance (paper (3)+(4) combined) ----------
    # z_t - w_t = sum_i (d - ghat + sc - eta_d sd - g)
    # Move variables left, constants right:
    # z_t - w_t - sum_i sc + sum_i eta_d sd + sum_i g = sum_i (d - ghat)
    energy_balance_row_ids = []  # to read duals later
    for tt in range(TT):
        row = np.zeros(n)
        row[flat("z", t=tt)] = 1
        row[flat("w", t=tt)] = -1

        for i, pl in enumerate(players):
            row[flat("sc", i=i, t=tt)] -= 1
            row[flat("sd", i=i, t=tt)] += params["eta_d"].get(pl, 1.0)
            row[flat("g",  i=i, t=tt)] += 1

        rhs = sum(params["d"][pl][tt] - params["ghat"][pl][tt] for pl in players)
        energy_balance_row_ids.append(len(A_eq))
        A_eq.append(row)
        b_eq.append(rhs)

    # ---------- PV limit (paper (5)) ----------
    # g[i,t] <= gtilde[i,t] * xG[i]
    for i, pl in enumerate(players):
        for tt in range(TT):
            row = np.zeros(n)
            row[flat("g", i=i, t=tt)] = 1
            row[flat("xG", i=i)] -= params["gtilde"][pl][tt]
            A_ub.append(row)
            b_ub.append(0.0)

    # ---------- BESS power limits (paper (6),(7)) ----------
    for i, pl in enumerate(players):
        schg = float(params["s_chg"].get(pl, 0.0))
        sdis = float(params["s_dis"].get(pl, 0.0))
        for tt in range(TT):
            # sc <= schg + xInv
            row = np.zeros(n)
            row[flat("sc", i=i, t=tt)] = 1
            row[flat("xInv", i=i)] -= 1
            A_ub.append(row); b_ub.append(schg)

            # sd <= sdis + xInv
            row = np.zeros(n)
            row[flat("sd", i=i, t=tt)] = 1
            row[flat("xInv", i=i)] -= 1
            A_ub.append(row); b_ub.append(sdis)

    # ---------- energy recursion (paper (9)) using u = e - Emin ----------
    # u[i,t] - u[i,t-1] - eta_c sc[i,t] + sd[i,t] = 0
    for i, pl in enumerate(players):
        eta_c = params["eta_c"].get(pl, 1.0)
        for tt in range(TT):
            row = np.zeros(n)
            row[flat("u", i=i, t=tt)] = 1
            if tt > 0:
                row[flat("u", i=i, t=tt-1)] -= 1
            row[flat("sc", i=i, t=tt)] -= eta_c
            row[flat("sd", i=i, t=tt)] += 1
            A_eq.append(row); b_eq.append(0.0)

        # Optional: fix initial energy to Emin (i.e., u[i,0]=0)
        if fix_initial_energy:
            row = np.zeros(n)
            row[flat("u", i=i, t=0)] = 1
            A_eq.append(row); b_eq.append(0.0)

    # ---------- energy upper bound (paper (8)) ----------
    # Emin <= e <= Ebar + xEn   ==> 0 <= u <= (Ebar - Emin) + xEn
    for i, pl in enumerate(players):
        cap = float(params["Ebar"].get(pl, 0.0) - params["Emin"].get(pl, 0.0))
        for tt in range(TT):
            row = np.zeros(n)
            row[flat("u", i=i, t=tt)] = 1
            row[flat("xEn", i=i)] -= 1
            A_ub.append(row); b_ub.append(cap)

    # ---------- end-of-horizon net zero usage (paper (10)) ----------
    # sum_t (eta_c sc - sd) = 0
    for i, pl in enumerate(players):
        eta_c = params["eta_c"].get(pl, 1.0)
        row = np.zeros(n)
        for tt in range(TT):
            row[flat("sc", i=i, t=tt)] = eta_c
            row[flat("sd", i=i, t=tt)] -= 1
        A_eq.append(row); b_eq.append(0.0)

    # ---------- solve ----------
    res = linprog(
        c,
        A_ub=np.array(A_ub), b_ub=np.array(b_ub),
        A_eq=np.array(A_eq), b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs"
    )
    if not res.success:
        raise RuntimeError(res.message)

    x = res.x

    def get(name):
        start, shape = idx[name]
        size = int(np.prod(shape))
        return x[start:start+size].reshape(shape)

    sol = {k: get(k) for k in idx.keys()}

    # Dual prices pi_t: marginals of energy-balance equalities
    duals_eq = res.eqlin.marginals
    pi = np.array([duals_eq[rid] for rid in energy_balance_row_ids], dtype=float)

    return res, sol, pi


# ------------------ DEMO TOY INSTANCE ------------------
players = ["A", "B", "C"]
T = [0, 1]  # 2 periods

params = {
    "cG": 3.0, "cInv": 10.0, "cEn": 5.0,
    "p": [4.0, 4.0],
    "h": [1.0, 1.0],

    "d": {
        "A": [2, 2],
        "B": [1, 1],
        "C": [1, 1],
    },
    "ghat": {  # existing PV
        "A": [2, 0],
        "B": [0, 2],
        "C": [0, 0],
    },
    "gtilde": {  # per-kW new PV yield
        "A": [1, 1],
        "B": [1, 1],
        "C": [1, 1],
    },

    # No existing battery in this demo
    "s_chg": {"A": 0, "B": 0, "C": 0},
    "s_dis": {"A": 0, "B": 0, "C": 0},
    "Emin":  {"A": 0, "B": 0, "C": 0},
    "Ebar":  {"A": 0, "B": 0, "C": 0},
    "eta_c": {"A": 1, "B": 1, "C": 1},
    "eta_d": {"A": 1, "B": 1, "C": 1},
}

res, sol, pi = solve_PS(players, T, params)

print("Optimal objective (C(S)) =", res.fun)
print("Dual local prices pi_t =", pi)
print("xG (new PV kW) =", sol["xG"])
print("z (buy) =", sol["z"], "w (sell) =", sol["w"])
