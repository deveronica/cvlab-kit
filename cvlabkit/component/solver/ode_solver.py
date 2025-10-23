"""ODE Solver using torchdiffeq."""

from __future__ import annotations

from torchdiffeq import odeint

from cvlabkit.component.base import Solver


class ODESolver(Solver):
    """ODE solver using adaptive step size integration.

    Wraps torchdiffeq for solving ordinary differential equations.
    Useful for continuous-time generative models, Neural ODEs, and flow matching.
    """

    def __init__(self, cfg):
        """Initialize ODE solver.

        Args:
            cfg: Configuration object with parameters:
                - method: Integration method (default: 'midpoint')
                  Options: 'dopri5', 'dopri8', 'bosh3', 'fehlberg2',
                          'adaptive_heun', 'euler', 'midpoint', 'rk4'
                - atol: Absolute tolerance (default: 1e-5)
                - rtol: Relative tolerance (default: 1e-5)
        """
        super().__init__()
        self.method = cfg.get("method", "midpoint")
        self.atol = cfg.get("atol", 1e-5)
        self.rtol = cfg.get("rtol", 1e-5)

    def __call__(self, ode_fn, initial_state, times):
        """Solve ODE: dx/dt = ode_fn(t, x).

        Args:
            ode_fn: Callable (t, x) -> dx/dt
            initial_state: Initial condition at times[0]
            times: Time points to evaluate (Tensor)

        Returns:
            Final state at times[-1]
        """
        trajectory = odeint(
            ode_fn,
            initial_state,
            times,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
        )
        return trajectory[-1]
