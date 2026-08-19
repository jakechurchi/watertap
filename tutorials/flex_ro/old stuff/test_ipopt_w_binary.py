import random

import pyomo.environ as pyo

from watertap.core.solvers import get_solver


def build_model():
    m = pyo.ConcreteModel()

    # 4 binary variables
    m.y = pyo.Var(range(1, 5), within=pyo.Binary, initialize=1)

    # 4 continuous variables
    m.x = pyo.Var(range(1, 5), bounds=(0, 5), initialize=1.0)

    # Add 2 constraints
    m.cons1 = pyo.Constraint(expr=sum(m.x[i] for i in m.x) >= 2.0)
    m.cons2 = pyo.Constraint(expr=m.x[1] + 0.5 * m.x[2] + 2.0 * m.y[1] - m.y[2] >= 1.0)

    # Nonlinear polynomial objective
    m.obj = pyo.Objective(
        expr=(m.x[1] - 1.3) ** 2
        + (m.x[2] - 0.8) ** 2
        + (m.x[3] - 1.5) ** 2
        + (m.x[4] - 0.7) ** 2
        + 0.2 * m.x[1] * m.x[3]
        + 0.05 * m.x[4] ** 3
        + 0.1 * (m.y[1] + m.y[2] + m.y[3] + m.y[4]),
        sense=pyo.minimize,
    )

    return m


def fix_binary_vars(m, seed=None):
    rng = random.Random(seed)
    for v in m.component_data_objects(pyo.Var, descend_into=True):
        if v.is_binary():
            v.fix(rng.randint(0, 1))


def solve_and_print(m):
    solver = get_solver()
    results = solver.solve(m, tee=False)

    print(f"Solver status: {results.solver.status}")
    print(f"Termination condition: {results.solver.termination_condition}")

    if (
        results.solver.status != pyo.SolverStatus.ok
        or results.solver.termination_condition != pyo.TerminationCondition.optimal
    ):
        raise RuntimeError("Solve did not converge to an optimal solution.")

    print("\nOptimal solution:")
    for i in m.x:
        print(f"x[{i}] = {pyo.value(m.x[i]):.6f}")
    for i in m.y:
        print(f"y[{i}] = {pyo.value(m.y[i]):.0f}")
    print(f"Objective = {pyo.value(m.obj):.6f}")


if __name__ == "__main__":
    model = build_model()
    fix_binary_vars(model)
    solve_and_print(model)
