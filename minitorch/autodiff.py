from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from collections import deque

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    shifted_vals = list(vals)
    shifted_vals[arg] += epsilon

    return (f(*shifted_vals) - f(*vals)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def dfs(current: Variable, used: set[int], result: List[Variable]):

    used.add(current.unique_id)
    for v in current.parents:
        if v.unique_id not in used:
            dfs(v, used, result)

    if not current.is_constant():
        result.append(current)


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    result = []
    used = set()
    dfs(variable, used, result)

    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    variable_dict: dict[int, Any] = {}
    variables_sorted = topological_sort(variable)[::-1]
    variables_sorted_ids = [v.unique_id for v in variables_sorted]
    variables_parents = [[v.parents for v in variables_sorted]]

    variable_dict[variable.unique_id] = deriv

    for v in variables_sorted:
        if v.is_leaf():
            v.accumulate_derivative(variable_dict[v.unique_id])
        else:
            for v, d in v.chain_rule(variable_dict[v.unique_id]):
                if v.unique_id not in variable_dict:
                    variable_dict[v.unique_id] = d
                else:
                    variable_dict[v.unique_id] += d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
