from dataclasses import dataclass


@dataclass
class Constraint:
    a: float
    b: float
    c: float
    name: str = ""

    def value(self, A: float, B: float) -> float:
        return self.a * A + self.b * B

    def is_satisfied(self, A: float, B: float, eps: float = 1e-9) -> bool:
        return self.value(A, B) <= self.c + eps


@dataclass
class OptimizationResult:
    maximum: float
    optimal_points: list
    candidates: list
    feasible_vertices: list


class LinearProgram2D:
    def __init__(
        self,
        objective_a: float,
        objective_b: float,
        constraints: list[Constraint],
    ):
        self.objective_a = objective_a
        self.objective_b = objective_b
        self.constraints = constraints

    def _boundaries(self):
        boundaries = list(self.constraints)

        boundaries.append(
            Constraint(-1, 0, 0, "A = 0")
        )

        boundaries.append(
            Constraint(0, -1, 0, "B = 0")
        )

        return boundaries

    def _intersection(self, first: Constraint, second: Constraint):
        a1, b1, c1 = first.a, first.b, first.c
        a2, b2, c2 = second.a, second.b, second.c

        D = a1 * b2 - a2 * b1

        if abs(D) < 1e-12:
            return None

        A = (c1 * b2 - c2 * b1) / D
        B = (a1 * c2 - a2 * c1) / D

        return A, B

    def _is_feasible(self, A: float, B: float) -> bool:
        eps = 1e-9

        if A < -eps or B < -eps:
            return False

        for constraint in self.constraints:
            if not constraint.is_satisfied(A, B):
                return False

        return True

    def _objective_value(self, A: float, B: float) -> float:
        return self.objective_a * A + self.objective_b * B

    def _unique_points(self, points):
        result = []

        for point in points:
            A, B = point

            already_exists = False

            for existing_A, existing_B in result:
                if (
                    abs(A - existing_A) < 1e-9
                    and abs(B - existing_B) < 1e-9
                ):
                    already_exists = True
                    break

            if not already_exists:
                result.append(point)

        return result

    def solve(self) -> OptimizationResult:
        boundaries = self._boundaries()

        candidates = []

        for i in range(len(boundaries)):
            for j in range(i + 1, len(boundaries)):
                point = self._intersection(
                    boundaries[i],
                    boundaries[j]
                )

                if point is not None:
                    candidates.append(point)

        candidates = self._unique_points(candidates)

        feasible_vertices = []

        for A, B in candidates:
            if self._is_feasible(A, B):
                feasible_vertices.append((A, B))

        feasible_vertices = self._unique_points(feasible_vertices)

        if not feasible_vertices:
            raise ValueError("Допустимых решений не найдено.")

        values = []

        for A, B in feasible_vertices:
            value = self._objective_value(A, B)
            values.append(value)

        maximum = max(values)

        optimal_points = []

        for point, value in zip(feasible_vertices, values):
            if abs(value - maximum) < 1e-9:
                optimal_points.append(point)

        return OptimizationResult(
            maximum=maximum,
            optimal_points=optimal_points,
            candidates=candidates,
            feasible_vertices=feasible_vertices,
        )


def print_result(result: OptimizationResult):
    print("Допустимые вершины:")

    for A, B in result.feasible_vertices:
        print(f"A = {A:.6f}, B = {B:.6f}")

    print()
    print(f"Максимальная прибыль: {result.maximum:.6f}")

    print("Оптимальные точки:")

    for A, B in result.optimal_points:
        print(f"A = {A:.6f}, B = {B:.6f}")


if __name__ == "__main__":
    constraints = [
        Constraint(6, 3, 80, "Древесина"),
        Constraint(2, 1, 40, "Фурнитура"),
        Constraint(1, 0, 15, "Максимум столов"),
        Constraint(0, 1, 30, "Максимум стульев"),
        Constraint(4, 2, 100, "Рабочее время"),
    ]

    problem = LinearProgram2D(
        objective_a=5000,
        objective_b=2500,
        constraints=constraints,
    )

    result = problem.solve()

    print_result(result)