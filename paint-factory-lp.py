import pyomo.environ as pyo

model = pyo.ConcreteModel()
model.x_1=pyo.Var(within=pyo.NonNegativeReals)
model.x_2=pyo.Var(within=pyo.NonNegativeReals)

model.obj = pyo.Objective(expr=5*model.x_1+4*model.x_2, sense=pyo.maximize)
model.con1=pyo.Constraint(expr=6*model.x_1+4*model.x_2<=24)
model.con2=pyo.Constraint(expr=model.x_1+2*model.x_2<=6)
model.con3=pyo.Constraint(expr=model.x_2-model.x_1<=1)
model.con4=pyo.Constraint(expr=model.x_2<=2)

# ---- SOLVE ----
solver = pyo.SolverFactory("highs")   # since you installed highspy
res = solver.solve(model, tee=False)

# ---- OUTPUT ----
print("status      :", res.solver.status)
print("termination :", res.solver.termination_condition)

print("x_1 =", pyo.value(model.x_1))
print("x_2 =", pyo.value(model.x_2))
print("obj =", pyo.value(model.obj))

# (optional) check each constraint body value
print("con1 LHS =", pyo.value(6*model.x_1 + 4*model.x_2))
print("con2 LHS =", pyo.value(model.x_1 + 2*model.x_2))
print("con3 LHS =", pyo.value(model.x_2 - model.x_1))
print("con4 LHS =", pyo.value(model.x_2))
