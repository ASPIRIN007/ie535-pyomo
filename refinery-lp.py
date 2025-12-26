import pyomo.environ as pyo

def build_and_solve():
    # ----------------------------
    # 1) DATA (parameters)
    # ----------------------------
    oils = ["light", "heavy"]
    products = ["gasoline", "kerosene", "jet"]

    cost = {"light": 20, "heavy": 15}
    loss = {"light": 0.05, "heavy": 0.08}

    # yields per barrel BEFORE loss (from the table)
    yield_raw = {
        ("light", "gasoline"): 0.40,
        ("light", "kerosene"): 0.20,
        ("light", "jet"):      0.35,
        ("heavy", "gasoline"): 0.32,
        ("heavy", "kerosene"): 0.40,
        ("heavy", "jet"):      0.20,
    }

    demand = {"gasoline": 1_000_000, "kerosene": 500_000, "jet": 300_000}

    # effective yields after loss
    y = {(o, p): (1 - loss[o]) * yield_raw[(o, p)] for o in oils for p in products}

    # ----------------------------
    # 2) MODEL
    # ----------------------------
    m = pyo.ConcreteModel()

    m.OILS = pyo.Set(initialize=oils)
    m.PRODS = pyo.Set(initialize=products)

    # decision variables: barrels of each crude
    m.x = pyo.Var(m.OILS, domain=pyo.NonNegativeReals)

    # objective: minimize cost
    m.obj = pyo.Objective(
        expr=sum(cost[o] * m.x[o] for o in m.OILS),
        sense=pyo.minimize
    )

    # constraints: meet demand for each product
    def demand_rule(m, p):
        return sum(y[(o, p)] * m.x[o] for o in m.OILS) >= demand[p]

    m.meet_demand = pyo.Constraint(m.PRODS, rule=demand_rule)

    # ----------------------------
    # 3) SOLVE
    # ----------------------------
    solver = pyo.SolverFactory("highs")
    if not solver.available():
        raise RuntimeError("HiGHS solver not available. Install highspy and try again.")

    res = solver.solve(m, tee=False)

    # ----------------------------
    # 4) PRINT RESULTS
    # ----------------------------
    print("Solver status:", res.solver.status)
    print("Termination  :", res.solver.termination_condition)
    print()

    x_light = pyo.value(m.x["light"])
    x_heavy = pyo.value(m.x["heavy"])
    print(f"x_light = {x_light:,.2f} barrels")
    print(f"x_heavy = {x_heavy:,.2f} barrels")
    print(f"Total cost = ${pyo.value(m.obj):,.2f}")
    print()

    # show produced amounts (to verify constraints)
    for p in products:
        produced = sum(y[(o, p)] * pyo.value(m.x[o]) for o in oils)
        print(f"{p:8s}: produced {produced:,.2f}  (demand {demand[p]:,})")

if __name__ == "__main__":
    build_and_solve()
