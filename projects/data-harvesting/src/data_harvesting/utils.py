import torch
from torchrl.modules.distributions.continuous import (
  FasterTransformedDistribution, Delta, constraints, is_compiling, _err_compile_safetanh, SafeTanhTransform,
  _PatchedAffineTransform, _PatchedComposeTransform, _cast_device
)
from numbers import Number


def collapse_shared_action_bound(bound: torch.Tensor | float) -> torch.Tensor | float:
    """Remove repeated agent rows, preserving bounds that differ between agents."""
    if isinstance(bound, torch.Tensor) and bound.ndim == 2 and bound.shape[0] > 0:
        per_action = bound[0]
        if torch.equal(bound, per_action.expand_as(bound)):
            return per_action.clone()
    return bound


def get_activation_class(name):
    if name == "Tanh":
        return torch.nn.Tanh
    elif name == "ReLU":
        return torch.nn.ReLU
    elif name == "LeakyReLU":
        return torch.nn.LeakyReLU
    else:
        raise ValueError(f"Unknown activation function: {name}")

def get_faster_tanh_delta(high: torch.Tensor | float, low: torch.Tensor | float) -> type[FasterTransformedDistribution]:
    """
    Runs pre-instantiation checks and returns a FasterTanhDelta class if all checks are passed.
    """
    minmax_msg = "high value has been found to be equal or less than low value"
    if isinstance(high, torch.Tensor) or isinstance(low, torch.Tensor):
        if is_compiling():
            assert (high > low).all()
        else:
            with torch.profiler.record_function("TanhDelta/gt"):
                high_gt_low = high > low
            with torch.profiler.record_function("TanhDelta/all"):
                high_gt_low_all = high_gt_low.all()
            if not high_gt_low_all:
                raise ValueError(minmax_msg)
    elif isinstance(high, Number) and isinstance(low, Number):
        if is_compiling():
            assert high > low
        elif high <= low:
            raise ValueError(minmax_msg)
    else:
        if not all(high > low):
            raise ValueError(minmax_msg)
    non_trivial_min = is_compiling or (
        (isinstance(low, torch.Tensor) and (low != -1.0).any())
        or (not isinstance(low, torch.Tensor) and low != -1.0)
    )
    non_trivial_max = is_compiling or (
        (isinstance(high, torch.Tensor) and (high != 1.0).any())
        or (not isinstance(high, torch.Tensor) and high != 1.0)
    )
    if non_trivial_min or non_trivial_max:
        return _NonTrivialFasterTanhDelta
    return _TrivialFasterTanhDelta

class _FasterTanhDelta(FasterTransformedDistribution):
    """Implements a Tanh transformed_in Delta distribution.

    This is an override of the default TanhDelta distribution from TorchRL which avoids synchronizing calls by
    following a fixed, known, path.

    Args:
        param (torch.Tensor): parameter of the delta distribution;
        low (torch.Tensor or number, optional): minimum value of the distribution. Default is -1.0;
        high (torch.Tensor or number, optional): maximum value of the distribution. Default is 1.0;
        event_dims (int, optional): number of dimensions describing the action.
            Default is 1;
        atol (number, optional): absolute tolerance to consider that a tensor matches the distribution parameter;
            Default is 1e-6
        rtol (number, optional): relative tolerance to consider that a tensor matches the distribution parameter;
            Default is 1e-6
        batch_shape (torch.Size, optional): batch shape;
        event_shape (torch.Size, optional): shape of the outcome;

    """

    arg_constraints = {
        "loc": constraints.real,
    }

    def __init__(
        self,
        param: torch.Tensor,
        low: torch.Tensor | float = -1.0,
        high: torch.Tensor | float = 1.0,
        event_dims: int = 1,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        safe: bool = True,
        non_trivial: bool = False,
    ):
        if safe:
            if is_compiling():
                _err_compile_safetanh()
            t = SafeTanhTransform()
        else:
            t = torch.distributions.TanhTransform()
        self.non_trivial = non_trivial

        self.low = _cast_device(low, param.device)
        self.high = _cast_device(high, param.device)
        loc = self.update(param)

        if self.non_trivial:
            t = _PatchedComposeTransform(
                [
                    t,
                    _PatchedAffineTransform(
                        loc=(self.high + self.low) / 2, scale=(self.high - self.low) / 2
                    ),
                ]
            )
        event_shape = param.shape[-event_dims:]
        batch_shape = param.shape[:-event_dims]
        base = Delta(
            loc,
            atol=atol,
            rtol=rtol,
            batch_shape=batch_shape,
            event_shape=event_shape,
        )

        super().__init__(base, t)

    @property
    def min(self):
        self._warn_minmax()
        return self.low

    @property
    def max(self):
        self._warn_minmax()
        return self.high

    def update(self, net_output: torch.Tensor) -> torch.Tensor | None:
        loc = net_output
        if self.non_trivial:
            device = loc.device
            shift = _cast_device(self.high - self.low, device)
            loc = loc + shift / 2 + _cast_device(self.low, device)
        if hasattr(self, "base_dist"):
            self.base_dist.update(loc)
        else:
            return loc

    @property
    def mode(self) -> torch.Tensor:
        mode = self.base_dist.param
        for t in self.transforms:
            mode = t(mode)
        return mode

    @property
    def deterministic_sample(self):
        return self.mode

    @property
    def mean(self) -> torch.Tensor:
        raise AttributeError("TanhDelta mean has not analytical form.")

class _NonTrivialFasterTanhDelta(_FasterTanhDelta):
    """FasterTanhDelta supporting non-trivial min/max values."""

    def __init__(
        self,
        param: torch.Tensor,
        low: torch.Tensor | float = -1.0,
        high: torch.Tensor | float = 1.0,
        event_dims: int = 1,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        safe: bool = True,
    ):
        super().__init__(
            param,
            low=low,
            high=high,
            event_dims=event_dims,
            atol=atol,
            rtol=rtol,
            safe=safe,
            non_trivial=True,
        )

class _TrivialFasterTanhDelta(_FasterTanhDelta):
    """FasterTanhDelta supporting only trivial min/max values (-1, 1)."""

    def __init__(
        self,
        param: torch.Tensor,
        event_dims: int = 1,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        safe: bool = True,
    ):
        super().__init__(
            param,
            low=-1.0,
            high=1.0,
            event_dims=event_dims,
            atol=atol,
            rtol=rtol,
            safe=safe,
            non_trivial=False,
        )
