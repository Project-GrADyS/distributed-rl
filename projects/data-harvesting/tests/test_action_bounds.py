import pytest
import torch

from data_harvesting.utils import collapse_shared_action_bound, get_faster_tanh_delta


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_shared_bounds_preserve_action_values(batch_shape: tuple[int, ...]) -> None:
    # Different limits for each action component must survive normalization.
    low = torch.tensor([-2.0, 0.0]).repeat(8, 1)
    high = torch.tensor([2.0, 1.0]).repeat(8, 1)
    param = torch.randn(*batch_shape, 8, 2)
    distribution = get_faster_tanh_delta(high, low)

    original = distribution(param, low=low, high=high).mode
    normalized = distribution(
        param,
        low=collapse_shared_action_bound(low),
        high=collapse_shared_action_bound(high),
    ).mode

    torch.testing.assert_close(normalized, original, rtol=0, atol=0)


def test_bounds_that_differ_between_agents_are_preserved() -> None:
    bound = torch.tensor([[0.0, 0.0], [0.0, 0.5]])

    assert collapse_shared_action_bound(bound) is bound
