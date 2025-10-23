from __future__ import annotations

from abc import abstractmethod

from cvlabkit.core.interface_meta import InterfaceMeta


class Solver(metaclass=InterfaceMeta):
    """Abstract base class for differential equation solvers.

    Solvers are used for ODE/SDE integration in generative models,
    neural ODEs, and other continuous-time systems.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Solve differential equation.

        Args:
            *args, **kwargs: Solver-specific inputs

        Returns:
            Solution of the differential equation
        """
        pass
